use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::GateError;

pub const DEFAULT_TEXT_MODEL_URL: &str =
    "https://huggingface.co/Xenova/gte-small/resolve/main/onnx/model_quantized.onnx";
pub const DEFAULT_TEXT_TOK_URL: &str =
    "https://huggingface.co/Xenova/gte-small/resolve/main/tokenizer.json";

pub const DEFAULT_NLI_MODEL_URL: &str =
    "https://huggingface.co/Xenova/nli-deberta-v3-xsmall/resolve/main/onnx/model_quantized.onnx";
pub const DEFAULT_NLI_TOK_URL: &str =
    "https://huggingface.co/Xenova/nli-deberta-v3-xsmall/resolve/main/tokenizer.json";

pub const DEFAULT_VISION_MODEL_URL: &str =
    "https://huggingface.co/twn39/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX/resolve/main/onnx/model_int8.onnx";
pub const DEFAULT_VISION_TOK_URL: &str =
    "https://huggingface.co/twn39/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX/resolve/main/tokenizer.json";

pub(crate) fn cache_dir(override_dir: Option<&Path>) -> PathBuf {
    if let Some(p) = override_dir {
        return p.to_path_buf();
    }
    if let Ok(env_dir) = std::env::var("WEAVETAB_GATE_CACHE") {
        if !env_dir.trim().is_empty() {
            return PathBuf::from(env_dir);
        }
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("weavetab-gate")
}

pub(crate) fn ensure_file(dir: &Path, rel: &str, url: &str) -> Result<PathBuf, GateError> {
    let dest = dir.join(rel);
    if dest.exists() && dest.metadata().map(|m| m.len() > 0).unwrap_or(false) {
        return Ok(dest);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp = dest.with_extension("part");
    let res = ureq::get(url)
        .set("User-Agent", "weavetab-gate/0.1")
        .call()
        .map_err(|e| GateError::ModelDownload(format!("{url}: {e}")))?;

    let mut reader = res.into_reader();
    let mut file = fs::File::create(&tmp)?;
    io::copy(&mut reader, &mut file)?;
    drop(file);
    fs::rename(&tmp, &dest)?;
    Ok(dest)
}
