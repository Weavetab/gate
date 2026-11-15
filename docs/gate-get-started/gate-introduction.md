---
title: "Introduction"
product: gate
section: gate-get-started
order: 1
description: "Local-first multimodal zero-shot decision engine in Rust + INT8 ONNX for Node.js, Python, Rust, and C."
---

# Weavetab Gate

**Weavetab Gate** is a local-first multimodal decision engine (`weavetab-gate` / `@weavetab/gate`). Give it **text** or an **image** (`PNG`, `JPEG`, `WebP`) alongside a bounded list of candidate choices, and it returns calibrated probabilities in milliseconds on CPU — with **zero LLM token generation**, **zero PyTorch**, and **zero API keys**.

## Why Gate Exists

Most classification workflows either force developers to pay per-token LLM fees for simple routing/triage, or install 2–3 GB of Python + PyTorch + `transformers` just to run a small encoder.

Gate compiles one Rust core (`crates/gate-core`) against ONNX Runtime (`ort`) and exposes native bindings for **TypeScript/Node.js**, **Python**, **Rust**, and **C/C++/Go/Java**:

| Component | Upstream Model | Format & Size | License |
|---|---|---|---|
| **Text Encoder** | `knowledgator/gliclass-edge-v3.0` | INT8 ONNX (`33.4 MB`) | Apache-2.0 |
| **Vision Encoder** | `TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` | INT8 ONNX (`24 MB`) | MIT |

Weights are lazy-downloaded once on first call into `~/.cache/weavetab-gate/` (configurable via `WEAVETAB_GATE_CACHE`).

## Measured CPU Benchmarks

Measured locally on Windows x64 (`debug` Rust build + ONNX Runtime C++, `threads = 2`):

| Workload | Uncached (ONNX CPU) | Cached (`RingCache`) | Top Confidence |
|---|---|---|---|
| **Image (solid color `224x224` PNG)** | `129 ms` | `0 ms` | `0.997` |
| **Image (`34.4 KB` wide taskbar PNG)** | `277 ms` | `0 ms` | `0.733` |
| **Text (`gliclass-edge-v3.0` INT8)** | `361 ms` | `0 ms` | `0.612` |
