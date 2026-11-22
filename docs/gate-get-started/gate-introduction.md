---
title: "Introduction"
product: gate
section: gate-get-started
order: 1
description: "Local-first multimodal zero-shot decision engine in Rust + INT8 ONNX for Node.js, Python, Rust, and C."
---

# Weavetab Gate

**Weavetab Gate** is a local-first multimodal decision engine (`weavetab-gate` / `@weavetab/gate`) built in Rust on top of ONNX Runtime (`ort`). Give it **text** or **images** (`PNG`, `JPEG`, `WebP`) alongside choices, a binary statement, or an ordered rubric — and get calibrated probabilities in milliseconds on CPU with **zero LLM token generation**, **zero PyTorch**, **zero API keys**, and **zero content-policy refusals**.

## Decision Primitives

| Primitive | Goal | Returns |
|---|---|---|
| **Yes/No** (`yes_no` / `yes_no_batch`) | Check whether a statement holds (NLI entailment) | `answer`, `probability` (`0.01`–`0.99`), `confidence`, `cached`, `latency_ms` |
| **Choice** (`decide` / `decide_batch`) | Pick one option from 2–64 choices (up to 10,000 inputs/call) | `choice`, `confidence`, `scores`, `cached`, `latency_ms` |
| **Score** (`score` / `score_batch`) | Rate input against ordered rubric levels | `level`, `expected`, `confidence`, `legend`, `scores`, `cached`, `latency_ms` |

## Engine Architecture

Gate lazy-loads each quantized ONNX engine only when that modality or primitive is called:

| Engine | Used By | Upstream Model | License |
|---|---|---|---|
| **Unified Text NLI Cross-Encoder** | `decide`, `yes_no`, `score` (+ `_batch`) | `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` (`72.2 MB` quantized ONNX) | MIT |
| **Vision Encoder** | Image inputs (`PNG` / `JPEG` / `WebP`) | `TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` (INT8 ONNX) | MIT |

Weights cache once in `~/.cache/weavetab-gate/` (or `%LOCALAPPDATA%\weavetab-gate` on Windows), overridable via `WEAVETAB_GATE_CACHE`.

## Why Gate vs Cloud Decision APIs

- **Up to 10,000 inputs in one call**: Length-bucketed `[B, seq]` micro-batching (`B = 16`), in-batch deduplication, and `RingCache` pre-pass keep peak RAM flat even on 4–8 GB laptops while reserving 1 CPU core for the OS.
- **3.7×–8.9× faster end-to-end on CPU**: `54.2–109.0 ms/item` locally vs `313.3–701.5 ms/item` over cloud RTT (and `5.0–8.2 ms/item` steady-state on Tesla T4 GPU).
- **Zero content-policy blocks**: Cloud APIs return `400 prohibited_content` when you try to classify abusive, toxic, or phishing messages for moderation. Gate runs 100% locally and scores every input.
- **\$0 per call**: No per-token billing, no RPM/TPM rate limits, works offline.
