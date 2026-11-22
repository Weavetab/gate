use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::models::{ensure_file, DEFAULT_NLI_MODEL_URL, DEFAULT_NLI_TOK_URL};
use crate::GateError;

pub(crate) struct NliEngine {
    session: Session,
    tokenizer: Tokenizer,
    has_type_ids: bool,
}

impl NliEngine {
    pub fn load(
        env: &ort::environment::Environment,
        cache_dir: &Path,
        threads: usize,
    ) -> Result<Self, GateError> {
        let model_url = std::env::var("WEAVETAB_GATE_NLI_URL")
            .unwrap_or_else(|_| DEFAULT_NLI_MODEL_URL.to_string());
        let tok_url = std::env::var("WEAVETAB_GATE_NLI_TOK_URL")
            .unwrap_or_else(|_| DEFAULT_NLI_TOK_URL.to_string());

        let model_path = ensure_file(cache_dir, "nli/gate_nli_i8.onnx", &model_url)?;
        let tok_path = ensure_file(cache_dir, "nli/gate_nli_tok.json", &tok_url)?;

        let session = Session::builder(env)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_path)?;
        let has_type_ids = session.inputs().iter().any(|i| i.name() == "token_type_ids");

        let mut tokenizer = Tokenizer::from_file(tok_path)
            .map_err(|e| GateError::Tokenizer(e.to_string()))?;
        tokenizer.with_padding(None);

        Ok(Self {
            session,
            tokenizer,
            has_type_ids,
        })
    }

    // returns entailment probability clamped to [0.01, 0.99] for each (premise, hypothesis) pair
    pub fn entailment_batch(&mut self, texts: &[&str], hypothesis: &str) -> Result<Vec<f32>, GateError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let pairs: Vec<(String, String)> = texts
            .iter()
            .map(|t| ((*t).to_string(), hypothesis.to_string()))
            .collect();

        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| GateError::Tokenizer(e.to_string()))?;

        let mut order: Vec<usize> = (0..encodings.len()).collect();
        order.sort_by_key(|&i| encodings[i].get_ids().len());

        let mut probs = vec![0.5f32; texts.len()];

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
            if raw.len() < b * 3 {
                return Err(GateError::Inference(format!(
                    "expected {} NLI logits, got {}",
                    b * 3,
                    raw.len()
                )));
            }

            for (row, &idx) in chunk.iter().enumerate() {
                // nli-deberta-v3-xsmall: 0=contradiction, 1=entailment, 2=neutral
                let c = raw[row * 3];
                let e = raw[row * 3 + 1];
                let m = c.max(e);
                let ec = (c - m).exp();
                let ee = (e - m).exp();
                let p = (ee / (ec + ee)).clamp(0.01, 0.99);
                probs[idx] = p;
            }
        }

        Ok(probs)
    }
}
