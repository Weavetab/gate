---
title: "SDK Architecture & Config"
product: gate
section: gate-api
order: 2
description: "Isolated per-language SDK layout (Node.js, Python, C ABI) and environment overrides."
---

# SDK Architecture & Configuration

`Weavetab/gate` separates the core Rust engine (`crates/gate-core`) from every language binding (`sdk/`), so adding or updating an SDK never touches inference logic:

```text
crates/gate-core/   # Stable Rust + ort + tokenizers core
sdk/node/           # @weavetab/gate (napi-rs)
sdk/python/         # weavetab-gate (PyO3 + maturin)
sdk/c/              # weavetab_gate.h C ABI (.dll / .so / .dylib) for C, C++, Go, Java
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `WEAVETAB_GATE_CACHE` | Override default `~/.cache/weavetab-gate/` weight directory |
| `WEAVETAB_GATE_TEXT_URL` | Custom text ONNX model download URL |
| `WEAVETAB_GATE_TEXT_TOK_URL` | Custom text `tokenizer.json` URL |
| `WEAVETAB_GATE_VISION_URL` | Custom vision ONNX model download URL |
| `WEAVETAB_GATE_VISION_TOK_URL` | Custom vision `tokenizer.json` URL |
