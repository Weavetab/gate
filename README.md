# Weavetab Gate

**Local-first multimodal zero-shot decision engine powered by Rust + INT8 ONNX.**

Feed **text** or **images** alongside arbitrary **choices** (`decide`, `yes_no`, `score`) and get calibrated probabilities in milliseconds on CPU — **no LLM token generation, no PyTorch, no API key.**

[![CI](https://github.com/Weavetab/gate/actions/workflows/ci.yml/badge.svg)](https://github.com/Weavetab/gate/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

---

## Install

| Language | Package | Install |
| --- | --- | --- |
| **Node.js / TypeScript** | `@weavetab/gate` | `npm install @weavetab/gate` |
| **Python** | `weavetab-gate` | `pip install weavetab-gate` |
| **Rust** | `weavetab-gate` | `cargo add weavetab-gate` |
| **C / C++ / Go / Java** | `weavetab_gate` | Build `sdk/c` (`weavetab_gate.h`) |

> Model weights (INT8 ONNX) are downloaded lazily on first run and cached locally.

---

## Quickstart

### TypeScript / Node.js (`@weavetab/gate`)

```ts
import fs from "node:fs";
import { Gate } from "@weavetab/gate";

const gate = new Gate();

// Text -> choice
const action = gate.decide("payment webhook returned 502 bad gateway", [
  "incident",
  "billing",
  "spam",
]);
// { choice: "incident", confidence: 0.94, scores: [...], cached: false, latencyMs: 12 }

// Image -> choice
const img = fs.readFileSync("screen.png");
const ui = gate.decide(img, ["login form", "captcha", "dashboard"]);

// Yes / No & Rubric Score
const urgent = gate.yesNo("server crashed with OOM", "is this a production outage?");
const severity = gate.score("minor typo in footer", ["low", "medium", "high", "critical"]);
```

### Python (`weavetab-gate`)

```python
from weavetab_gate import Gate

gate = Gate()

res = gate.decide("refund requested after 45 days", ["approve", "escalate", "deny"])
print(res["choice"], res["confidence"])

with open("receipt.png", "rb") as f:
    img_res = gate.decide(f.read(), ["valid receipt", "blurry", "spam"])

urgent = gate.yes_no("server crashed with OOM", "is this a production outage?")
severity = gate.score("minor typo in footer", ["low", "medium", "high", "critical"])
```

### Rust (`weavetab-gate`)

```rust
use weavetab_gate::{Gate, Input};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gate = Gate::default();
    let d = gate.decide(Input::Text("hello"), &["greeting", "question", "complaint"])?;
    println!("{} ({:.2})", d.choice, d.confidence);
    Ok(())
}
```

---

## API Overview

| Method | Input | Output |
| --- | --- | --- |
| `decide(input, choices)` | Text (`str`/`string`) or image bytes (`bytes`/`Buffer`) + label list | Winning `choice`, `confidence`, full `scores` distribution, `cached`, `latency_ms` |
| `yes_no(input, question)` | Text or image bytes + boolean question | `answer` (`bool`), `probability`, `cached`, `latency_ms` |
| `score(input, levels)` | Text or image bytes + ordered rubric labels | Top `level`, `expected` index score, full `scores`, `cached`, `latency_ms` |

---

## Architecture

- **`crates/gate-core`** (`weavetab-gate` on crates.io): Core Rust runtime built on ONNX Runtime (`ort`), pure-Rust `tokenizers`, `image`, lazy Hugging Face weight cache, and a `4096`-entry in-memory `RingCache`.
- **`sdk/node`** (`@weavetab/gate` on npm): Native Node.js / TypeScript binding via `napi-rs`.
- **`sdk/python`** (`weavetab-gate` on PyPI): Native Python wheel via `PyO3` + `maturin` (zero PyTorch).
- **`sdk/c`** (`weavetab_gate`): C ABI shared/static library ([`weavetab_gate.h`](sdk/c/include/weavetab_gate.h)) for C, C++, Go (`cgo`), and Java.

---

## License & Upstream Models

Licensed under **Apache-2.0**. See [NOTICE](NOTICE) for full upstream attribution:

- **Text encoder**: `knowledgator/gliclass-edge-v3.0` (Apache-2.0, INT8 ONNX `33.4 MB`)
- **Vision encoder**: `TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` (MIT, INT8 ONNX `24 MB`)

Questions or security reports: [contact@weavetab.dev](mailto:contact@weavetab.dev) · [weavetab.dev](https://weavetab.dev)
