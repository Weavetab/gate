use std::path::PathBuf;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use weavetab_gate::{Config as CoreConfig, Gate as CoreGate, Input};

#[napi(object)]
pub struct GateConfig {
    pub cache_dir: Option<String>,
    pub cache_entries: Option<u32>,
    pub threads: Option<u32>,
}

#[napi(object)]
pub struct OptionScore {
    pub label: String,
    pub score: f64,
}

#[napi(object)]
pub struct Decision {
    pub choice: String,
    pub confidence: f64,
    pub scores: Vec<OptionScore>,
    pub cached: bool,
    pub latency_ms: f64,
}

#[napi(object)]
pub struct YesNo {
    pub answer: bool,
    pub probability: f64,
    pub confidence: f64,
    pub cached: bool,
    pub latency_ms: f64,
}

#[napi(object)]
pub struct Score {
    pub level: String,
    pub expected: f64,
    pub confidence: f64,
    pub legend: Vec<String>,
    pub scores: Vec<OptionScore>,
    pub cached: bool,
    pub latency_ms: f64,
}

#[napi]
pub struct Gate {
    inner: CoreGate,
}

fn to_err(e: weavetab_gate::GateError) -> Error {
    Error::from_reason(e.to_string())
}

fn map_scores(scores: Vec<weavetab_gate::OptionScore>) -> Vec<OptionScore> {
    scores
        .into_iter()
        .map(|s| OptionScore {
            label: s.label,
            score: s.score as f64,
        })
        .collect()
}

fn with_input<T>(
    input: Either<String, Buffer>,
    f: impl FnOnce(Input<'_>) -> std::result::Result<T, weavetab_gate::GateError>,
) -> Result<T> {
    match input {
        Either::A(text) => f(Input::Text(&text)).map_err(to_err),
        Either::B(buf) => f(Input::ImageBytes(buf.as_ref())).map_err(to_err),
    }
}

#[napi]
impl Gate {
    #[napi(constructor)]
    pub fn new(config: Option<GateConfig>) -> Self {
        let mut cfg = CoreConfig::default();
        if let Some(c) = config {
            if let Some(dir) = c.cache_dir {
                cfg.cache_dir = Some(PathBuf::from(dir));
            }
            if let Some(cap) = c.cache_entries {
                cfg.cache_entries = cap as usize;
            }
            if let Some(t) = c.threads {
                cfg.threads = t as usize;
            }
        }
        Self {
            inner: CoreGate::new(cfg),
        }
    }

    #[napi]
    pub fn cache_path(&self) -> String {
        self.inner.cache_path().to_string_lossy().into_owned()
    }

    #[napi]
    pub fn decide(&self, input: Either<String, Buffer>, choices: Vec<String>) -> Result<Decision> {
        let refs: Vec<&str> = choices.iter().map(String::as_str).collect();
        let d = with_input(input, |inp| self.inner.decide(inp, &refs))?;
        Ok(Decision {
            choice: d.choice,
            confidence: d.confidence as f64,
            scores: map_scores(d.scores),
            cached: d.cached,
            latency_ms: d.latency_ms as f64,
        })
    }

    #[napi]
    pub fn yes_no(&self, input: Either<String, Buffer>, question: String) -> Result<YesNo> {
        let r = with_input(input, |inp| self.inner.yes_no(inp, &question))?;
        Ok(YesNo {
            answer: r.answer,
            probability: r.probability as f64,
            confidence: r.confidence as f64,
            cached: r.cached,
            latency_ms: r.latency_ms as f64,
        })
    }

    #[napi]
    pub fn score(&self, input: Either<String, Buffer>, levels: Vec<String>) -> Result<Score> {
        let refs: Vec<&str> = levels.iter().map(String::as_str).collect();
        let s = with_input(input, |inp| self.inner.score(inp, &refs))?;
        Ok(Score {
            level: s.level,
            expected: s.expected as f64,
            confidence: s.confidence as f64,
            legend: s.legend,
            scores: map_scores(s.scores),
            cached: s.cached,
            latency_ms: s.latency_ms as f64,
        })
    }

    #[napi]
    pub fn decide_batch(
        &self,
        inputs: Vec<Either<String, Buffer>>,
        choices: Vec<String>,
    ) -> Result<Vec<Decision>> {
        let refs: Vec<&str> = choices.iter().map(String::as_str).collect();
        let core_inputs: Vec<Input<'_>> = inputs
            .iter()
            .map(|inp| match inp {
                Either::A(s) => Input::Text(s.as_str()),
                Either::B(b) => Input::ImageBytes(b.as_ref()),
            })
            .collect();
        let list = self
            .inner
            .decide_batch(&core_inputs, &refs)
            .map_err(to_err)?;
        Ok(list
            .into_iter()
            .map(|d| Decision {
                choice: d.choice,
                confidence: d.confidence as f64,
                scores: map_scores(d.scores),
                cached: d.cached,
                latency_ms: d.latency_ms as f64,
            })
            .collect())
    }

    #[napi]
    pub fn yes_no_batch(
        &self,
        inputs: Vec<Either<String, Buffer>>,
        question: String,
    ) -> Result<Vec<YesNo>> {
        let core_inputs: Vec<Input<'_>> = inputs
            .iter()
            .map(|inp| match inp {
                Either::A(s) => Input::Text(s.as_str()),
                Either::B(b) => Input::ImageBytes(b.as_ref()),
            })
            .collect();
        let list = self
            .inner
            .yes_no_batch(&core_inputs, &question)
            .map_err(to_err)?;
        Ok(list
            .into_iter()
            .map(|r| YesNo {
                answer: r.answer,
                probability: r.probability as f64,
                confidence: r.confidence as f64,
                cached: r.cached,
                latency_ms: r.latency_ms as f64,
            })
            .collect())
    }

    #[napi]
    pub fn score_batch(
        &self,
        inputs: Vec<Either<String, Buffer>>,
        levels: Vec<String>,
    ) -> Result<Vec<Score>> {
        let refs: Vec<&str> = levels.iter().map(String::as_str).collect();
        let core_inputs: Vec<Input<'_>> = inputs
            .iter()
            .map(|inp| match inp {
                Either::A(s) => Input::Text(s.as_str()),
                Either::B(b) => Input::ImageBytes(b.as_ref()),
            })
            .collect();
        let list = self
            .inner
            .score_batch(&core_inputs, &refs)
            .map_err(to_err)?;
        Ok(list
            .into_iter()
            .map(|s| Score {
                level: s.level,
                expected: s.expected as f64,
                confidence: s.confidence as f64,
                legend: s.legend,
                scores: map_scores(s.scores),
                cached: s.cached,
                latency_ms: s.latency_ms as f64,
            })
            .collect())
    }
}
