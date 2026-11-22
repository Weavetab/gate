use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::models::{ensure_file, DEFAULT_TEXT_MODEL_URL, DEFAULT_TEXT_TOK_URL};
use crate::{GateError, OptionScore};

const DIM: usize = 384;
const TEMP: f32 = 0.05;

pub(crate) struct TextEngine {
    session: Session,
    tokenizer: Tokenizer,
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

        let model_path = ensure_file(cache_dir, "text/gate_v1_i8.onnx", &model_url)?;
        let tok_path = ensure_file(cache_dir, "text/gate_v1_tok.json", &tok_url)?;

        let session = Session::builder(env)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_path)?;
        let mut tokenizer = Tokenizer::from_file(tok_path)
            .map_err(|e| GateError::Tokenizer(e.to_string()))?;
        // disable baked-in fixed-512 padding from HF tokenizer.json; we pad per micro-batch dynamically
        tokenizer.with_padding(None);

        Ok(Self { session, tokenizer })
    }

    // bi-encoder CLS embeddings, L2-normalized, length-bucketed in B=32 micro-batches
    fn embed_batch(&mut self, items: &[String]) -> Result<Vec<[f32; DIM]>, GateError> {
        let encodings = self
            .tokenizer
            .encode_batch(items.to_vec(), true)
            .map_err(|e| GateError::Tokenizer(e.to_string()))?;

        let mut order: Vec<usize> = (0..encodings.len()).collect();
        order.sort_by_key(|&i| encodings[i].get_ids().len());

        let mut embeds = vec![[0.0f32; DIM]; items.len()];

        // ponytail: single-session sequential micro-batches keep peak RAM flat on low-end PCs
        for chunk in order.chunks(32) {
            let b = chunk.len();
            let max_len = chunk
                .iter()
                .map(|&i| encodings[i].get_ids().len().min(512))
                .max()
                .unwrap_or(1);

            let mut ids = vec![0i64; b * max_len];
            let mut mask = vec![0i64; b * max_len];
            let type_ids = vec![0i64; b * max_len];

            for (row, &idx) in chunk.iter().enumerate() {
                let enc_ids = encodings[idx].get_ids();
                let enc_mask = encodings[idx].get_attention_mask();
                let n = enc_ids.len().min(max_len);
                let off = row * max_len;
                for col in 0..n {
                    ids[off + col] = enc_ids[col] as i64;
                    mask[off + col] = enc_mask[col] as i64;
                }
            }

            let ids_t = Tensor::from_array(([b, max_len], ids))?;
            let mask_t = Tensor::from_array(([b, max_len], mask))?;
            let type_t = Tensor::from_array(([b, max_len], type_ids))?;

            let outputs = self.session.run(ort::inputs![
                "input_ids" => ids_t,
                "attention_mask" => mask_t,
                "token_type_ids" => type_t,
            ])?;

            let (_shape, raw) = outputs[0].try_extract_tensor::<f32>()?;
            let row_stride = max_len * DIM;
            if raw.len() < b * row_stride {
                return Err(GateError::Inference(format!(
                    "expected {} floats, got {}",
                    b * row_stride,
                    raw.len()
                )));
            }

            for (row, &idx) in chunk.iter().enumerate() {
                // GTE-small uses attention-masked mean pooling + L2 norm
                let enc_mask = encodings[idx].get_attention_mask();
                let n = enc_mask.len().min(max_len);
                let mut sum = [0.0f32; DIM];
                let mut count = 0.0f32;
                let base = row * row_stride;
                for tok in 0..n {
                    if enc_mask[tok] != 0 {
                        let tok_off = base + tok * DIM;
                        for d in 0..DIM {
                            sum[d] += raw[tok_off + d];
                        }
                        count += 1.0;
                    }
                }
                let denom = count.max(1.0);
                for d in 0..DIM {
                    sum[d] /= denom;
                }
                let norm = sum.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
                for d in 0..DIM {
                    sum[d] /= norm;
                }
                embeds[idx] = sum;
            }
        }

        Ok(embeds)
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

        // encode choices ONCE for the entire batch (up to 10,000 texts)
        let choice_strs: Vec<String> = choices.iter().map(|c| (*c).to_string()).collect();
        let choice_embeds = self.embed_batch(&choice_strs)?;

        let text_strs: Vec<String> = texts.iter().map(|t| (*t).to_string()).collect();
        let text_embeds = self.embed_batch(&text_strs)?;

        let mut results = Vec::with_capacity(texts.len());
        for te in &text_embeds {
            let mut sims = Vec::with_capacity(choices.len());
            let mut max_s = f32::NEG_INFINITY;
            for ce in &choice_embeds {
                let mut dot = 0.0f32;
                for d in 0..DIM {
                    dot += te[d] * ce[d];
                }
                let scaled = dot / TEMP;
                if scaled > max_s {
                    max_s = scaled;
                }
                sims.push(scaled);
            }

            let exps: Vec<f32> = sims.iter().map(|&s| (s - max_s).exp()).collect();
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
}
