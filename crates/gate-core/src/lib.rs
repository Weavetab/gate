mod cache;
mod models;
mod text;
mod vision;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use cache::{hash_key, RingCache};
use models::cache_dir;
use text::TextEngine;
use vision::VisionEngine;

#[derive(Debug, Error)]
pub enum GateError {
    #[error("provide between 2 and 64 choices")]
    InvalidChoices,
    #[error("provide between 1 and 10000 inputs")]
    InvalidBatch,
    #[error("empty input")]
    EmptyInput,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model download failed: {0}")]
    ModelDownload(String),
    #[error("onnx runtime error: {0}")]
    Ort(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("image decode error: {0}")]
    Image(String),
    #[error("inference error: {0}")]
    Inference(String),
}

impl<R> From<ort::Error<R>> for GateError {
    fn from(e: ort::Error<R>) -> Self {
        Self::Ort(e.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub cache_dir: Option<PathBuf>,
    pub cache_entries: usize,
    pub threads: usize,
}

impl Default for Config {
    fn default() -> Self {
        // leave 1 core free so 10k batches don't freeze low-end laptops
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        Self {
            cache_dir: None,
            cache_entries: 4096,
            threads: cpus.saturating_sub(1).clamp(1, 4),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Input<'a> {
    Text(&'a str),
    ImageBytes(&'a [u8]),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionScore {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub choice: String,
    pub confidence: f32,
    pub scores: Vec<OptionScore>,
    pub cached: bool,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YesNo {
    pub answer: bool,
    pub probability: f32,
    pub confidence: f32,
    pub cached: bool,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub level: String,
    pub expected: f32,
    pub confidence: f32,
    pub legend: Vec<String>,
    pub scores: Vec<OptionScore>,
    pub cached: bool,
    pub latency_ms: u64,
}

struct Inner {
    env: Option<ort::environment::Environment>,
    text: Option<TextEngine>,
    vision: Option<VisionEngine>,
    cache: RingCache<Vec<OptionScore>>,
}

pub struct Gate {
    dir: PathBuf,
    threads: usize,
    inner: Mutex<Inner>,
}

impl Gate {
    pub fn new(cfg: Config) -> Self {
        Self {
            dir: cache_dir(cfg.cache_dir.as_deref()),
            threads: cfg.threads.max(1),
            inner: Mutex::new(Inner {
                env: None,
                text: None,
                vision: None,
                cache: RingCache::new(cfg.cache_entries),
            }),
        }
    }

    pub fn cache_path(&self) -> &Path {
        &self.dir
    }

    pub fn decide(&self, input: Input<'_>, choices: &[&str]) -> Result<Decision, GateError> {
        if choices.len() < 2 || choices.len() > 64 {
            return Err(GateError::InvalidChoices);
        }
        let (tag, payload): (u8, &[u8]) = match input {
            Input::Text(s) => {
                if s.trim().is_empty() {
                    return Err(GateError::EmptyInput);
                }
                (1, s.as_bytes())
            }
            Input::ImageBytes(b) => {
                if b.is_empty() {
                    return Err(GateError::EmptyInput);
                }
                (2, b)
            }
        };

        let key = hash_key(tag, payload, choices);
        let t0 = Instant::now();

        let mut guard = self.inner.lock().expect("gate mutex poisoned");
        if let Some(scores) = guard.cache.get(key) {
            let top = &scores[0];
            return Ok(Decision {
                choice: top.label.clone(),
                confidence: top.score,
                scores,
                cached: true,
                latency_ms: t0.elapsed().as_millis() as u64,
            });
        }

        if guard.env.is_none() {
            guard.env = Some(ort::init().with_name("weavetab-gate").build()?);
        }
        let env = guard.env.as_ref().unwrap().clone();

        let scores = match input {
            Input::Text(s) => {
                if guard.text.is_none() {
                    guard.text = Some(TextEngine::load(&env, &self.dir, self.threads)?);
                }
                guard.text.as_mut().unwrap().classify(s, choices)?
            }
            Input::ImageBytes(b) => {
                if guard.vision.is_none() {
                    guard.vision = Some(VisionEngine::load(&env, &self.dir, self.threads)?);
                }
                guard.vision.as_mut().unwrap().classify(b, choices)?
            }
        };

        guard.cache.put(key, scores.clone());
        let top = &scores[0];
        Ok(Decision {
            choice: top.label.clone(),
            confidence: top.score,
            scores,
            cached: false,
            latency_ms: t0.elapsed().as_millis() as u64,
        })
    }

    pub fn yes_no(&self, input: Input<'_>, question: &str) -> Result<YesNo, GateError> {
        let mut out = self.yes_no_batch(&[input], question)?;
        Ok(out.pop().expect("single yes_no resolved"))
    }

    pub fn score(&self, input: Input<'_>, levels: &[&str]) -> Result<Score, GateError> {
        let d = self.decide(input, levels)?;
        let expected = levels
            .iter()
            .enumerate()
            .map(|(i, lvl)| {
                let p = d
                    .scores
                    .iter()
                    .find(|s| s.label == *lvl)
                    .map(|s| s.score)
                    .unwrap_or(0.0);
                (i as f32) * p
            })
            .sum();
        Ok(Score {
            level: d.choice,
            expected,
            confidence: d.confidence,
            legend: levels.iter().map(|s| (*s).to_string()).collect(),
            scores: d.scores,
            cached: d.cached,
            latency_ms: d.latency_ms,
        })
    }

    pub fn decide_batch(
        &self,
        inputs: &[Input<'_>],
        choices: &[&str],
    ) -> Result<Vec<Decision>, GateError> {
        if choices.len() < 2 || choices.len() > 64 {
            return Err(GateError::InvalidChoices);
        }
        if inputs.is_empty() || inputs.len() > 10_000 {
            return Err(GateError::InvalidBatch);
        }

        let mut keys = Vec::with_capacity(inputs.len());
        for &inp in inputs {
            let (tag, payload): (u8, &[u8]) = match inp {
                Input::Text(s) => {
                    if s.trim().is_empty() {
                        return Err(GateError::EmptyInput);
                    }
                    (1, s.as_bytes())
                }
                Input::ImageBytes(b) => {
                    if b.is_empty() {
                        return Err(GateError::EmptyInput);
                    }
                    (2, b)
                }
            };
            keys.push(hash_key(tag, payload, choices));
        }

        let t0 = Instant::now();
        let mut guard = self.inner.lock().expect("gate mutex poisoned");

        let mut out: Vec<Option<Decision>> = vec![None; inputs.len()];
        let mut seen_text: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        let mut uniq_texts: Vec<&str> = Vec::new();
        let mut text_targets: Vec<Vec<usize>> = Vec::new();
        let mut img_indices: Vec<usize> = Vec::new();

        for (i, &key) in keys.iter().enumerate() {
            if let Some(scores) = guard.cache.get(key) {
                let top = &scores[0];
                out[i] = Some(Decision {
                    choice: top.label.clone(),
                    confidence: top.score,
                    scores,
                    cached: true,
                    latency_ms: 0,
                });
                continue;
            }
            match inputs[i] {
                Input::Text(s) => {
                    if let Some(&u) = seen_text.get(&key) {
                        text_targets[u].push(i);
                    } else {
                        let u = uniq_texts.len();
                        seen_text.insert(key, u);
                        uniq_texts.push(s);
                        text_targets.push(vec![i]);
                    }
                }
                Input::ImageBytes(_) => img_indices.push(i),
            }
        }

        if (!uniq_texts.is_empty() || !img_indices.is_empty()) && guard.env.is_none() {
            guard.env = Some(ort::init().with_name("weavetab-gate").build()?);
        }

        if !uniq_texts.is_empty() {
            let env = guard.env.as_ref().unwrap().clone();
            if guard.text.is_none() {
                guard.text = Some(TextEngine::load(&env, &self.dir, self.threads)?);
            }
            let batch_scores = guard.text.as_mut().unwrap().classify_batch(&uniq_texts, choices)?;
            for (u, scores) in batch_scores.into_iter().enumerate() {
                let first_idx = text_targets[u][0];
                guard.cache.put(keys[first_idx], scores.clone());
                let top = &scores[0];
                for &orig_i in &text_targets[u] {
                    out[orig_i] = Some(Decision {
                        choice: top.label.clone(),
                        confidence: top.score,
                        scores: scores.clone(),
                        cached: false,
                        latency_ms: 0,
                    });
                }
            }
        }

        for i in img_indices {
            if let Some(scores) = guard.cache.get(keys[i]) {
                let top = &scores[0];
                out[i] = Some(Decision {
                    choice: top.label.clone(),
                    confidence: top.score,
                    scores,
                    cached: true,
                    latency_ms: 0,
                });
                continue;
            }
            let Input::ImageBytes(b) = inputs[i] else { continue };
            let env = guard.env.as_ref().unwrap().clone();
            if guard.vision.is_none() {
                guard.vision = Some(VisionEngine::load(&env, &self.dir, self.threads)?);
            }
            let scores = guard.vision.as_mut().unwrap().classify(b, choices)?;
            guard.cache.put(keys[i], scores.clone());
            let top = &scores[0];
            out[i] = Some(Decision {
                choice: top.label.clone(),
                confidence: top.score,
                scores,
                cached: false,
                latency_ms: 0,
            });
        }

        let elapsed = t0.elapsed().as_millis() as u64;
        Ok(out
            .into_iter()
            .map(|d| {
                let mut dec = d.expect("all batch items resolved");
                dec.latency_ms = elapsed;
                dec
            })
            .collect())
    }

    pub fn yes_no_batch(
        &self,
        inputs: &[Input<'_>],
        question: &str,
    ) -> Result<Vec<YesNo>, GateError> {
        if inputs.is_empty() || inputs.len() > 10_000 {
            return Err(GateError::InvalidBatch);
        }
        if question.trim().is_empty() {
            return Err(GateError::EmptyInput);
        }

        let mut keys = Vec::with_capacity(inputs.len());
        for &inp in inputs {
            let (tag, payload): (u8, &[u8]) = match inp {
                Input::Text(s) => {
                    if s.trim().is_empty() {
                        return Err(GateError::EmptyInput);
                    }
                    (3, s.as_bytes())
                }
                Input::ImageBytes(b) => {
                    if b.is_empty() {
                        return Err(GateError::EmptyInput);
                    }
                    (4, b)
                }
            };
            keys.push(hash_key(tag, payload, &[question]));
        }

        let t0 = Instant::now();
        let mut guard = self.inner.lock().expect("gate mutex poisoned");

        let mut out: Vec<Option<YesNo>> = vec![None; inputs.len()];
        let mut seen_text: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        let mut uniq_texts: Vec<&str> = Vec::new();
        let mut text_targets: Vec<Vec<usize>> = Vec::new();
        let mut img_indices: Vec<usize> = Vec::new();

        for (i, &key) in keys.iter().enumerate() {
            if let Some(scores) = guard.cache.get(key) {
                let p = scores.first().map(|s| s.score).unwrap_or(0.5);
                out[i] = Some(YesNo {
                    answer: p >= 0.5,
                    probability: p,
                    confidence: p.max(1.0 - p),
                    cached: true,
                    latency_ms: 0,
                });
                continue;
            }
            match inputs[i] {
                Input::Text(s) => {
                    if let Some(&u) = seen_text.get(&key) {
                        text_targets[u].push(i);
                    } else {
                        let u = uniq_texts.len();
                        seen_text.insert(key, u);
                        uniq_texts.push(s);
                        text_targets.push(vec![i]);
                    }
                }
                Input::ImageBytes(_) => img_indices.push(i),
            }
        }

        if (!uniq_texts.is_empty() || !img_indices.is_empty()) && guard.env.is_none() {
            guard.env = Some(ort::init().with_name("weavetab-gate").build()?);
        }

        if !uniq_texts.is_empty() {
            let env = guard.env.as_ref().unwrap().clone();
            if guard.text.is_none() {
                guard.text = Some(TextEngine::load(&env, &self.dir, self.threads)?);
            }
            let probs = guard
                .text
                .as_mut()
                .unwrap()
                .yes_no_batch(&uniq_texts, question)?;
            for (u, p) in probs.into_iter().enumerate() {
                let first_idx = text_targets[u][0];
                guard.cache.put(
                    keys[first_idx],
                    vec![OptionScore {
                        label: "yes".into(),
                        score: p,
                    }],
                );
                for &orig_i in &text_targets[u] {
                    out[orig_i] = Some(YesNo {
                        answer: p >= 0.5,
                        probability: p,
                        confidence: p.max(1.0 - p),
                        cached: false,
                        latency_ms: 0,
                    });
                }
            }
        }

        if !img_indices.is_empty() {
            let yes = format!("yes: {question}");
            let no = format!("no: {question}");
            for i in img_indices {
                let Input::ImageBytes(b) = inputs[i] else { continue };
                let env = guard.env.as_ref().unwrap().clone();
                if guard.vision.is_none() {
                    guard.vision = Some(VisionEngine::load(&env, &self.dir, self.threads)?);
                }
                let scores = guard.vision.as_mut().unwrap().classify(b, &[&yes, &no])?;
                let p = scores
                    .iter()
                    .find(|s| s.label == yes)
                    .map(|s| s.score)
                    .unwrap_or(0.5)
                    .clamp(0.01, 0.99);
                guard.cache.put(
                    keys[i],
                    vec![OptionScore {
                        label: "yes".into(),
                        score: p,
                    }],
                );
                out[i] = Some(YesNo {
                    answer: p >= 0.5,
                    probability: p,
                    confidence: p.max(1.0 - p),
                    cached: false,
                    latency_ms: 0,
                });
            }
        }

        let elapsed = t0.elapsed().as_millis() as u64;
        Ok(out
            .into_iter()
            .map(|r| {
                let mut yn = r.expect("all yes_no items resolved");
                yn.latency_ms = elapsed;
                yn
            })
            .collect())
    }

    pub fn score_batch(
        &self,
        inputs: &[Input<'_>],
        levels: &[&str],
    ) -> Result<Vec<Score>, GateError> {
        let decisions = self.decide_batch(inputs, levels)?;
        let legend: Vec<String> = levels.iter().map(|s| (*s).to_string()).collect();
        Ok(decisions
            .into_iter()
            .map(|d| {
                let expected = levels
                    .iter()
                    .enumerate()
                    .map(|(i, lvl)| {
                        let p = d
                            .scores
                            .iter()
                            .find(|s| s.label == *lvl)
                            .map(|s| s.score)
                            .unwrap_or(0.0);
                        (i as f32) * p
                    })
                    .sum();
                Score {
                    level: d.choice,
                    expected,
                    confidence: d.confidence,
                    legend: legend.clone(),
                    scores: d.scores,
                    cached: d.cached,
                    latency_ms: d.latency_ms,
                }
            })
            .collect())
    }
}

impl Default for Gate {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_inputs() {
        let g = Gate::default();
        assert!(matches!(
            g.decide(Input::Text("hi"), &["only_one"]),
            Err(GateError::InvalidChoices)
        ));
        assert!(matches!(
            g.decide(Input::Text("   "), &["a", "b"]),
            Err(GateError::EmptyInput)
        ));
        assert!(matches!(
            g.decide(Input::ImageBytes(&[]), &["a", "b"]),
            Err(GateError::EmptyInput)
        ));
        assert!(matches!(
            g.decide_batch(&[], &["a", "b"]),
            Err(GateError::InvalidBatch)
        ));
        let too_many = vec![Input::Text("hi"); 10_001];
        assert!(matches!(
            g.decide_batch(&too_many, &["a", "b"]),
            Err(GateError::InvalidBatch)
        ));
    }

    #[test]
    fn cache_evicts_at_cap() {
        let mut c = RingCache::new(2);
        c.put(1, "a");
        c.put(2, "b");
        c.put(3, "c");
        assert_eq!(c.get(1), None);
        assert_eq!(c.get(2), Some("b"));
        assert_eq!(c.get(3), Some("c"));
    }

    #[test]
    fn preprocesses_tiny_png() {
        let img = image::RgbImage::from_pixel(16, 12, image::Rgb([255, 0, 128]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let tensor = vision::preprocess_image(buf.get_ref()).unwrap();
        assert_eq!(tensor.len(), 3 * 224 * 224);
    }
}
