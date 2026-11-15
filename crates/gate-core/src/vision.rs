use std::path::Path;

use image::imageops::FilterType;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::models::{ensure_file, DEFAULT_VISION_MODEL_URL, DEFAULT_VISION_TOK_URL};
use crate::{GateError, OptionScore};

const CLIP_SIZE: u32 = 224;
const CLIP_SEQ: usize = 77;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub(crate) struct VisionEngine {
    session: Session,
    tokenizer: Tokenizer,
}

impl VisionEngine {
    pub fn load(
        env: &ort::environment::Environment,
        cache_dir: &Path,
        threads: usize,
    ) -> Result<Self, GateError> {
        let model_url = std::env::var("WEAVETAB_GATE_VISION_URL")
            .unwrap_or_else(|_| DEFAULT_VISION_MODEL_URL.to_string());
        let tok_url = std::env::var("WEAVETAB_GATE_VISION_TOK_URL")
            .unwrap_or_else(|_| DEFAULT_VISION_TOK_URL.to_string());

        let model_path = ensure_file(cache_dir, "vision/model_int8.onnx", &model_url)?;
        let tok_path = ensure_file(cache_dir, "vision/tokenizer.json", &tok_url)?;

        let session = Session::builder(env)?
            .with_intra_threads(threads)?
            .commit_from_file(model_path)?;
        let tokenizer = Tokenizer::from_file(tok_path)
            .map_err(|e| GateError::Tokenizer(e.to_string()))?;

        Ok(Self { session, tokenizer })
    }

    pub fn classify(
        &mut self,
        img_bytes: &[u8],
        choices: &[&str],
    ) -> Result<Vec<OptionScore>, GateError> {
        let pixels = preprocess_image(img_bytes)?;
        let n = choices.len();

        let mut all_ids = Vec::with_capacity(n * CLIP_SEQ);
        let mut all_mask = Vec::with_capacity(n * CLIP_SEQ);

        for &choice in choices {
            let enc = self
                .tokenizer
                .encode(choice, true)
                .map_err(|e| GateError::Tokenizer(e.to_string()))?;
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            for i in 0..CLIP_SEQ {
                all_ids.push(ids.get(i).copied().unwrap_or(0) as i64);
                all_mask.push(mask.get(i).copied().unwrap_or(0) as i64);
            }
        }

        let ids_t = Tensor::from_array(([n, CLIP_SEQ], all_ids))?;
        let mask_t = Tensor::from_array(([n, CLIP_SEQ], all_mask))?;
        let pix_t = Tensor::from_array(([1usize, 3, CLIP_SIZE as usize, CLIP_SIZE as usize], pixels))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
            "pixel_values" => pix_t,
        ])?;

        // TinyCLIP exports logits_per_image as first output [1, N]
        let (_shape, raw) = outputs[0].try_extract_tensor::<f32>()?;
        let logits: Vec<f32> = raw.iter().take(n).copied().collect();
        if logits.len() != n {
            return Err(GateError::Inference(format!(
                "expected {n} vision logits, got {}",
                logits.len()
            )));
        }

        let probs = softmax(&logits);
        let mut scored: Vec<OptionScore> = choices
            .iter()
            .zip(probs)
            .map(|(&label, score)| OptionScore {
                label: label.to_string(),
                score,
            })
            .collect();
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(scored)
    }
}

pub(crate) fn preprocess_image(bytes: &[u8]) -> Result<Vec<f32>, GateError> {
    let img = image::load_from_memory(bytes).map_err(|e| GateError::Image(e.to_string()))?;
    let (w, h) = (img.width(), img.height());
    let side = w.min(h);
    let cropped = img.crop_imm((w - side) / 2, (h - side) / 2, side, side);
    let rgb = cropped
        .resize_exact(CLIP_SIZE, CLIP_SIZE, FilterType::CatmullRom)
        .to_rgb8();

    let plane = (CLIP_SIZE * CLIP_SIZE) as usize;
    let mut chw = vec![0.0f32; 3 * plane];
    for (idx, p) in rgb.pixels().enumerate() {
        chw[idx] = (p[0] as f32 / 255.0 - MEAN[0]) / STD[0];
        chw[plane + idx] = (p[1] as f32 / 255.0 - MEAN[1]) / STD[1];
        chw[2 * plane + idx] = (p[2] as f32 / 255.0 - MEAN[2]) / STD[2];
    }
    Ok(chw)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_v = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-9);
    exps.into_iter().map(|e| e / sum).collect()
}
