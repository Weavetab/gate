# Weavetab Gate

Local-first multimodal zero-shot decision engine powered by Rust + INT8 ONNX.

Feed **text** or **images** + a list of **choices** (`decide`, `yes_no`, `score`) and get calibrated probabilities in milliseconds on CPU — **no LLM token generation, no PyTorch, no API key.**

## Architecture

- **`crates/gate-core`** (`weavetab-gate` on crates.io): Stable Rust core (ONNX Runtime via `ort`, pure-Rust `tokenizers`, `image`, lazy Hugging Face weight cache, and `4096`-entry RAM `RingCache`).
- **`sdk/node`** (`@weavetab/gate` on npm): Native Node.js / TypeScript binding via `napi-rs`.
- **`sdk/python`** (`weavetab-gate` on PyPI): Native Python binding via `PyO3` + `maturin` (zero PyTorch).
- **`sdk/c`** (`weavetab_gate`): C ABI shared/static library (`weavetab_gate.h`) for C, C++, Go (`cgo`), and Java.

## Quickstart

### TypeScript / Node.js (`@weavetab/gate`)

```ts
import fs from "node:fs";
import { Gate } from "@weavetab/gate";

const gate = new Gate();

// Text -> choice
const action = gate.decide("player has 2 hearts and creeper nearby", [
  "retreat",
  "attack",
  "eat food",
]);

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
img_res = gate.decide(open("receipt.png", "rb").read(), ["valid receipt", "blurry", "spam"])
```

### Rust (`weavetab-gate`)

```rust
use weavetab_gate::{Gate, Input};

let gate = Gate::default();
let d = gate.decide(Input::Text("hello"), &["greeting", "question", "complaint"])?;
```

## License & Upstream Models

Licensed under **Apache-2.0**. See [NOTICE](NOTICE) for full upstream attribution:
- Text encoder: `knowledgator/gliclass-edge-v3.0` (Apache-2.0, INT8 ONNX `33.4 MB`)
- Vision encoder: `TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` (MIT, INT8 ONNX `24 MB`)
