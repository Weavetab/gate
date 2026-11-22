import os
import urllib.request
from pathlib import Path

import onnxruntime as ort
from huggingface_hub import HfApi

SOURCES = {
    "text/model_i8.onnx": "https://huggingface.co/MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33/resolve/main/onnx/model_quantized.onnx",
    "text/tokenizer.json": "https://huggingface.co/MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33/resolve/main/tokenizer.json",
    "vision/model_i8.onnx": "https://huggingface.co/twn39/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX/resolve/main/onnx/model_int8.onnx",
    "vision/tokenizer.json": "https://huggingface.co/twn39/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX/resolve/main/tokenizer.json",
}

MODEL_CARD = """---
license: apache-2.0
language:
  - en
library_name: weavetab-gate
tags:
  - zero-shot-classification
  - onnx
  - int8
  - rust
  - multimodal
  - weavetab
---

# Weavetab Gate v1.1 (`weavetab/gate-1`)

Quantized INT8 ONNX weights for **[Weavetab Gate](https://github.com/Weavetab/gate)** — a local-first multimodal zero-shot decision engine in Rust with native Node.js (`@weavetab/gate`) and Python (`weavetab-gate`) bindings.

- **GitHub**: [Weavetab/gate](https://github.com/Weavetab/gate)
- **npm**: `npm install @weavetab/gate`
- **PyPI**: `pip install weavetab-gate`
- **crates.io**: `cargo add weavetab-gate`
- **Website**: [weavetab.dev](https://weavetab.dev) · [contact@weavetab.dev](mailto:contact@weavetab.dev)

## Attribution & License

Licensed under **Apache-2.0**. See `NOTICE` for full upstream model attribution:
- Unified text zero-shot NLI encoder derived from `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` (MIT)
- Vision encoder derived from `wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` (MIT)
"""

OUT = Path("dist_weights")


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def optimize_onnx(src: Path, dst: Path) -> None:
    opts = ort.SessionOptions()
    # ORT_ENABLE_BASIC is portable across x64 + arm64 (ORT_ENABLE_ALL bakes host-specific NCHWC ops)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(dst)
    sess = ort.InferenceSession(str(src), sess_options=opts, providers=["CPUExecutionProvider"])
    print(
        dst.name,
        "inputs:", [(i.name, i.shape) for i in sess.get_inputs()],
        "outputs:", [(o.name, o.shape) for o in sess.get_outputs()],
    )


def main() -> None:
    raw = OUT / "raw"
    opt = OUT / "opt"
    for rel, url in SOURCES.items():
        download(url, raw / rel)
        dst = opt / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel == "vision/model_i8.onnx":
            optimize_onnx(raw / rel, dst)
        else:
            dst.write_bytes((raw / rel).read_bytes())

    for doc in ["../LICENSE", "../NOTICE"]:
        p = Path(__file__).resolve().parent / doc
        if p.exists():
            (opt / p.name).write_bytes(p.read_bytes())
    (opt / "README.md").write_text(MODEL_CARD, encoding="utf-8")

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    api.create_repo(repo_id="weavetab/gate-1", repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(opt),
        repo_id="weavetab/gate-1",
        repo_type="model",
    )
    print("Uploaded optimized weights to https://huggingface.co/weavetab/gate-1")


if __name__ == "__main__":
    main()
