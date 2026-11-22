---
title: "SDK Architecture & Config"
product: gate
section: gate-api
order: 2
description: "Multi-language SDK layout, GateConfig options, low-end PC thread guardrails, and environment overrides."
---

# SDK Architecture & Configuration

`Weavetab/gate` isolates the core Rust engine (`crates/gate-core`) from every language binding (`sdk/`), so all languages share identical inference, caching, and batching behavior:

```text
crates/gate-core/   # Rust + ONNX Runtime (ort) + tokenizers + RingCache
sdk/node/           # @weavetab/gate (napi-rs native .node binaries)
sdk/python/         # weavetab-gate (PyO3 + maturin native wheels)
sdk/c/              # weavetab_gate.h C ABI (.dll / .so / .dylib) for C, C++, Go, Java
```

## Configuration (`GateConfig`)

| Option | Type | Default | Description |
|---|---|---|---|
| `cache_dir` (`cacheDir`) | `string \| null` | OS cache dir | Directory for cached ONNX weights and tokenizers |
| `cache_entries` (`cacheEntries`) | `number` | `4096` | Max entries in the in-memory `RingCache` |
| `threads` | `number` | `clamp(cores - 1, 1, 4)` | ONNX Runtime intra-op CPU threads (leaves 1 core free for OS responsiveness on low-end laptops) |

## Environment Variables

| Variable | Purpose |
|---|---|
| `WEAVETAB_GATE_CACHE` | Override default `~/.cache/weavetab-gate/` weight directory |
| `WEAVETAB_GATE_TEXT_URL` | Custom unified text NLI ONNX model URL |
| `WEAVETAB_GATE_TEXT_TOK_URL` | Custom unified text `tokenizer.json` URL |
| `WEAVETAB_GATE_VISION_URL` | Custom vision ONNX model URL |
| `WEAVETAB_GATE_VISION_TOK_URL` | Custom vision `tokenizer.json` URL |
