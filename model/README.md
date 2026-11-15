---
license: apache-2.0
tags:
  - zero-shot-classification
  - text-classification
  - onnx
  - int8
  - decision
  - weavetab
base_model:
  - knowledgator/gliclass-edge-v3.0
  - cnmoro/gliclass-edge-v3.0-onnx
---

# Weavetab Gate 1 (`weavetab/gate-1`)

Quantized INT8 ONNX single-pass decision model powering **Weavetab Gate** (`https://api.weavetab.dev`).

## Files
* `model_i8.onnx` (33.4 MB) — INT8 quantized ONNX weights
* `tokenizer.json` (3.58 MB) — ModernBERT tokenizer

## Attribution & License
Licensed under **Apache License 2.0**, derived from `knowledgator/gliclass-edge-v3.0` (Knowledgator) and `cnmoro/gliclass-edge-v3.0-onnx`.
The AGPL-3.0 API gateway source code lives at [github.com/Weavetab/gate](https://github.com/Weavetab/gate).
