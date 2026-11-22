use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::models::{ensure_file, DEFAULT_TEXT_MODEL_URL, DEFAULT_TEXT_TOK_URL};
use crate::{GateError, OptionScore};

const TEMP: f32 = 1.5;

pub(crate) struct TextEngine {
    session: Session,
    tokenizer: Tokenizer,
    has_type_ids: bool,
}

impl TextEngine {
    pub fn load(
        env: &ort::environment::Environment,
        cache_dir: &Path,
        threads: usize,
    ) -> Result<Self, GateError> {
        let model_url = std::env::var("WEAVETAB_GATE_TEXT_URL")
            .unwrap_or_else(|_| DEFAULT_TEXT_MODEL_URL.to_string());
        let tok_url = std::env::var("WEAVETAB_GATE_TEXT_TOK_URL")
            .unwrap_or_else(|_| DEFAULT_TEXT_TOK_URL.to_string());

        let model_path = ensure_file(cache_dir, "text/gate_v11_i8.onnx", &model_url)?;
        let tok_path = ensure_file(cache_dir, "text/gate_v11_tok.json", &tok_url)?;

        let session = Session::builder(env)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_path)?;
        let has_type_ids = session.inputs().iter().any(|i| i.name() == "token_type_ids");

        let mut tokenizer = Tokenizer::from_file(tok_path)
            .map_err(|e| GateError::Tokenizer(e.to_string()))?;
        // disable baked-in padding; we pad dynamically per length-bucketed micro-batch
        tokenizer.with_padding(None);

        Ok(Self {
            session,
            tokenizer,
            has_type_ids,
        })
    }

    // returns entailment margin (logit_0 - logit_1) for each (premise, hypothesis) pair
    fn score_pairs(&mut self, pairs: Vec<(String, String)>) -> Result<Vec<f32>, GateError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| GateError::Tokenizer(e.to_string()))?;

        let mut order: Vec<usize> = (0..encodings.len()).collect();
        order.sort_by_key(|&i| encodings[i].get_ids().len());

        let mut margins = vec![0.0f32; encodings.len()];

        // ponytail: single-session sequential micro-batches keep peak RAM flat on low-end PCs
        for chunk in order.chunks(16) {
            let b = chunk.len();
            let max_len = chunk
                .iter()
                .map(|&i| encodings[i].get_ids().len().min(512))
                .max()
                .unwrap_or(1);

            let mut ids = vec![0i64; b * max_len];
            let mut mask = vec![0i64; b * max_len];
            let mut type_ids = vec![0i64; b * max_len];

            for (row, &idx) in chunk.iter().enumerate() {
                let enc_ids = encodings[idx].get_ids();
                let enc_mask = encodings[idx].get_attention_mask();
                let enc_types = encodings[idx].get_type_ids();
                let n = enc_ids.len().min(max_len);
                let off = row * max_len;
                for col in 0..n {
                    ids[off + col] = enc_ids[col] as i64;
                    mask[off + col] = enc_mask[col] as i64;
                    type_ids[off + col] = enc_types[col] as i64;
                }
            }

            let ids_t = Tensor::from_array(([b, max_len], ids))?;
            let mask_t = Tensor::from_array(([b, max_len], mask))?;

            let outputs = if self.has_type_ids {
                let type_t = Tensor::from_array(([b, max_len], type_ids))?;
                self.session.run(ort::inputs![
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t,
                    "token_type_ids" => type_t,
                ])?
            } else {
                self.session.run(ort::inputs![
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t,
                ])?
            };

            let (_shape, raw) = outputs[0].try_extract_tensor::<f32>()?;
            if raw.len() < b * 2 {
                return Err(GateError::Inference(format!(
                    "expected at least {} logits, got {}",
                    b * 2,
                    raw.len()
                )));
            }
            let stride = raw.len() / b;

            for (row, &idx) in chunk.iter().enumerate() {
                // MoritzLaurer zeroshot NLI: 0 = entailment, 1 = not_entailment
                margins[idx] = raw[row * stride] - raw[row * stride + 1];
            }
        }

        Ok(margins)
    }

    pub fn classify(&mut self, text: &str, choices: &[&str]) -> Result<Vec<OptionScore>, GateError> {
        let mut out = self.classify_batch(&[text], choices)?;
        Ok(out.pop().unwrap_or_default())
    }

    pub fn classify_batch(
        &mut self,
        texts: &[&str],
        choices: &[&str],
    ) -> Result<Vec<Vec<OptionScore>>, GateError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let k = choices.len();
        let hypotheses: Vec<String> = choices
            .iter()
            .map(|c| format!("This text is about {c}."))
            .collect();

        let mut pairs = Vec::with_capacity(texts.len() * k);
        for &t in texts {
            for h in &hypotheses {
                pairs.push((t.to_string(), h.clone()));
            }
        }

        let margins = self.score_pairs(pairs)?;
        let mut results = Vec::with_capacity(texts.len());

        for (i, _) in texts.iter().enumerate() {
            let slice = &margins[i * k..(i + 1) * k];
            let max_m = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = slice.iter().map(|&m| ((m - max_m) / TEMP).exp()).collect();
            let sum: f32 = exps.iter().sum::<f32>().max(1e-9);

            let mut scored: Vec<OptionScore> = choices
                .iter()
                .zip(exps)
                .map(|(&label, e)| OptionScore {
                    label: label.to_string(),
                    score: e / sum,
                })
                .collect();
            scored.sort_by(|a, b| b.score.total_cmp(&a.score));
            results.push(scored);
        }

        Ok(results)
    }

    pub fn yes_no_batch(&mut self, texts: &[&str], statement: &str) -> Result<Vec<f32>, GateError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let neg = format!("This is not true: {statement}");
        let mut pairs = Vec::with_capacity(texts.len() * 2);
        for &t in texts {
            pairs.push((t.to_string(), statement.to_string()));
            pairs.push((t.to_string(), neg.clone()));
        }

        let margins = self.score_pairs(pairs)?;
        let mut probs = Vec::with_capacity(texts.len());
        for i in 0..texts.len() {
            let m_yes = margins[i * 2];
            let m_no = margins[i * 2 + 1];
            let diff = ((m_yes - m_no) / TEMP).clamp(-12.0, 12.0);
            let p = (1.0 / (1.0 + (-diff).exp())).clamp(0.01, 0.99);
            probs.push(p);
        }
        Ok(probs)
    }
}
