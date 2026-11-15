use axum::{
    extract::{DefaultBodyLimit, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use gliclass::{input::text::TextInput, params::Parameters, GLiClass};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    env,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::Semaphore;
use tower_http::cors::{Any, CorsLayer};

const MAX_BODY_BYTES: usize = 4096;
const MAX_PROMPT_CHARS: usize = 1024;
const MIN_OPTIONS: usize = 2;
const MAX_OPTIONS: usize = 16;
const MAX_CONCURRENCY: usize = 64;
const CACHE_CAP: usize = 4096;

struct AppState {
    engine: GLiClass,
    gate: Arc<Semaphore>,
    cache: Mutex<RingCache>,
}

struct RingCache {
    map: HashMap<String, DecideOut>,
    order: VecDeque<String>,
}

impl RingCache {
    fn new() -> Self {
        Self {
            map: HashMap::with_capacity(CACHE_CAP),
            order: VecDeque::with_capacity(CACHE_CAP),
        }
    }

    fn get(&self, k: &str) -> Option<DecideOut> {
        self.map.get(k).cloned()
    }

    fn insert(&mut self, k: String, v: DecideOut) {
        if self.map.contains_key(&k) {
            return;
        }
        if self.order.len() >= CACHE_CAP {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
        self.order.push_back(k.clone());
        self.map.insert(k, v);
    }
}

#[derive(Deserialize)]
struct DecideBody {
    prompt: String,
    #[serde(default)]
    options: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct DecideQuery {
    q: String,
    #[serde(default)]
    options: Option<String>,
}

#[derive(Clone, Serialize)]
struct DecideOut {
    ok: bool,
    model: &'static str,
    decision: String,
    confidence: f32,
    scores: HashMap<String, f32>,
    cached: bool,
}

#[derive(Serialize)]
struct ErrEnvelope {
    ok: bool,
    error: ErrDetail,
}

#[derive(Serialize)]
struct ErrDetail {
    code: &'static str,
    message: String,
    docs: &'static str,
}

fn api_err(status: StatusCode, code: &'static str, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrEnvelope {
            ok: false,
            error: ErrDetail {
                code,
                message: message.into(),
                docs: "https://weavetab.dev/docs/gate/gate-errors",
            },
        }),
    )
        .into_response()
}

fn require_ua(headers: &HeaderMap) -> Result<(), Response> {
    let ok = headers
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    if !ok {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            "MISSING_USER_AGENT",
            "Include a non-empty User-Agent header (e.g. my-bot/1.0).",
        ));
    }
    Ok(())
}

fn normalize_options(raw: Option<Vec<String>>) -> Result<Vec<String>, Response> {
    let opts = raw.unwrap_or_else(|| vec!["yes".into(), "no".into()]);
    let cleaned: Vec<String> = opts
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if cleaned.len() < MIN_OPTIONS || cleaned.len() > MAX_OPTIONS {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            "INVALID_OPTIONS",
            format!("Provide between {MIN_OPTIONS} and {MAX_OPTIONS} non-empty options."),
        ));
    }
    Ok(cleaned)
}

async fn run_decide(state: Arc<AppState>, prompt: String, options: Vec<String>) -> Result<DecideOut, Response> {
    let prompt = prompt.trim().to_string();
    if prompt.is_empty() {
        return Err(api_err(StatusCode::BAD_REQUEST, "EMPTY_PROMPT", "prompt must not be empty."));
    }
    if prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            "INPUT_TOO_LONG",
            format!("prompt exceeds {MAX_PROMPT_CHARS} character limit."),
        ));
    }

    let cache_key = format!("{prompt}\x1f{}", options.join("\x1f"));
    if let Some(mut hit) = state.cache.lock().unwrap().get(&cache_key) {
        hit.cached = true;
        return Ok(hit);
    }

    let permit = state
        .gate
        .clone()
        .try_acquire_owned()
        .map_err(|_| api_err(StatusCode::SERVICE_UNAVAILABLE, "SERVER_BUSY", "inference queue full, retry shortly"))?;

    let st = state.clone();
    let opts_clone = options.clone();
    let res = tokio::time::timeout(
        Duration::from_secs(3),
        tokio::task::spawn_blocking(move || -> Result<DecideOut, String> {
            let _permit = permit;
            let opt_refs: Vec<&str> = opts_clone.iter().map(|s| s.as_str()).collect();
            let input = TextInput::from_str(&[prompt.as_str()], &opt_refs);
            let out = st.engine.inference(input).map_err(|e| e.to_string())?;

            let mut scores = HashMap::with_capacity(opts_clone.len());
            let mut best_label = opts_clone[0].clone();
            let mut best_score = f32::NEG_INFINITY;

            for (idx, label) in opts_clone.iter().enumerate() {
                let s = out.scores[[0, idx]];
                scores.insert(label.clone(), s);
                if s > best_score {
                    best_score = s;
                    best_label = label.clone();
                }
            }

            Ok(DecideOut {
                ok: true,
                model: "weavetab-gate-1",
                decision: best_label,
                confidence: best_score.clamp(0.0, 1.0),
                scores,
                cached: false,
            })
        }),
    )
    .await;

    match res {
        Ok(Ok(Ok(out))) => {
            state.cache.lock().unwrap().insert(cache_key, out.clone());
            Ok(out)
        }
        Ok(Ok(Err(e))) => Err(api_err(StatusCode::INTERNAL_SERVER_ERROR, "INFERENCE_FAILED", e)),
        Ok(Err(_)) => Err(api_err(StatusCode::INTERNAL_SERVER_ERROR, "WORKER_PANIC", "inference worker failed")),
        Err(_) => Err(api_err(StatusCode::GATEWAY_TIMEOUT, "INFERENCE_TIMEOUT", "inference exceeded 3s limit")),
    }
}

async fn post_decide(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<DecideBody>,
) -> Response {
    if let Err(e) = require_ua(&headers) {
        return e;
    }
    let options = match normalize_options(body.options) {
        Ok(o) => o,
        Err(e) => return e,
    };
    match run_decide(state, body.prompt, options).await {
        Ok(out) => (StatusCode::OK, Json(out)).into_response(),
        Err(e) => e,
    }
}

async fn get_decide(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<DecideQuery>,
) -> Response {
    if let Err(e) = require_ua(&headers) {
        return e;
    }
    let raw_opts = q.options.map(|s| s.split(',').map(|p| p.to_string()).collect());
    let options = match normalize_options(raw_opts) {
        Ok(o) => o,
        Err(e) => return e,
    };
    match run_decide(state, q.q, options).await {
        Ok(out) => (
            StatusCode::OK,
            [(header::CACHE_CONTROL, "public, max-age=3600")],
            Json(out),
        )
            .into_response(),
        Err(e) => e,
    }
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "ok": true,
        "model": "weavetab-gate-1",
        "status": "ready"
    }))
}

fn build_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any);
    Router::new()
        .route("/health", get(health))
        .route("/v1/decide", post(post_decide).get(get_decide))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(cors)
        .with_state(state)
}

#[tokio::main]
async fn main() {
    let model_path = env::var("GATE_MODEL_PATH").unwrap_or_else(|_| "model/model_i8.onnx".into());
    let tok_path = env::var("GATE_TOKENIZER_PATH").unwrap_or_else(|_| "model/tokenizer.json".into());
    let port: u16 = env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(7860);

    let engine = GLiClass::new(&tok_path, &model_path, Parameters::default())
        .expect("failed to load gate onnx model");

    let state = Arc::new(AppState {
        engine,
        gate: Arc::new(Semaphore::new(MAX_CONCURRENCY)),
        cache: Mutex::new(RingCache::new()),
    });

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("gate listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind failed");
    axum::serve(listener, build_router(state)).await.expect("serve failed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn decide_and_guards() {
        let engine = GLiClass::new("model/tokenizer.json", "model/model_i8.onnx", Parameters::default())
            .expect("model load");
        let state = Arc::new(AppState {
            engine,
            gate: Arc::new(Semaphore::new(MAX_CONCURRENCY)),
            cache: Mutex::new(RingCache::new()),
        });
        let app = build_router(state);

        let no_ua = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/decide")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(no_ua.status(), StatusCode::BAD_REQUEST);

        let ok_res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/decide")
                    .header("content-type", "application/json")
                    .header("user-agent", "gate-test/1.0")
                    .body(Body::from(r#"{"prompt":"Server CPU is at 99%","options":["scale_up","ignore"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ok_res.status(), StatusCode::OK);
    }
}
