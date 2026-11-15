use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::models::{ensure_file, DEFAULT_TEXT_MODEL_URL, DEFAULT_TEXT_TOK_URL};
use crate::{GateError, OptionScore};

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

        let model_path = ensure_file(cache_dir, "text/model_int8.onnx", &model_url)?;
        let tok_path = ensure_file(cache_dir, "text/tokenizer.json", &tok_url)?;

        let session = Session::builder(env)?
            .with_intra_threads(threads)?
            .commit_from_file(model_path)?;
        let tokenizer = Tokenizer::from_file(tok_path)
            .map_err(|e| GateError::Tokenizer(e.to_string()))?;

        Ok(Self { session, tokenizer })
    }

    pub fn classify(&mut self, text: &str, choices: &[&str]) -> Result<Vec<OptionScore>, GateError> {
        // GLiClass expects <<LABEL>>choice_1<<LABEL>>choice_2<<SEP>>text
        let mut prompt = String::with_capacity(text.len() + choices.len() * 24 + 16);
        for c in choices {
            prompt.push_str("<<LABEL>>");
            prompt.push_str(c);
        }
        prompt.push_str("<<SEP>>");
        prompt.push_str(text);

        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| GateError::Tokenizer(e.to_string()))?;

        let ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&m| m as i64).collect();
        let seq = ids.len();

        let ids_t = Tensor::from_array(([1usize, seq], ids))?;
        let mask_t = Tensor::from_array(([1usize, seq], mask))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
        ])?;

        let (_shape, raw) = outputs[0].try_extract_tensor::<f32>()?;
        let logits: Vec<f32> = raw.iter().take(choices.len()).copied().collect();
        if logits.len() != choices.len() {
            return Err(GateError::Inference(format!(
                "expected {} logits, got {}",
                choices.len(),
                logits.len()
            )));
        }

        // GLiClass outputs independent logits per label; sigmoid + L1 norm gives calibrated choice weights
        let probs: Vec<f32> = logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        let sum: f32 = probs.iter().sum::<f32>().max(1e-9);

        let mut scored: Vec<OptionScore> = choices
            .iter()
            .zip(probs)
            .map(|(&label, p)| OptionScore {
                label: label.to_string(),
                score: p / sum,
            })
            .collect();
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(scored)
    }
}
