---
title: "Quickstart"
product: gate
section: gate-get-started
order: 2
description: "Install and run Weavetab Gate in TypeScript/Node.js, Python, Rust, or C."
---

# Quickstart

## TypeScript / Node.js (`@weavetab/gate`)

```bash
npm install @weavetab/gate
```

```ts
import fs from "node:fs";
import { Gate } from "@weavetab/gate";

const gate = new Gate();

const triage = gate.decide("payment webhook returned 502 bad gateway", [
  "production incident",
  "billing inquiry",
  "feature request",
]);
console.log(triage.choice, triage.confidence, triage.latencyMs);

const screenshot = fs.readFileSync("screen.png");
const ui = gate.decide(screenshot, ["login form", "captcha", "dashboard"]);
console.log(ui.choice, ui.confidence);
```

## Python (`weavetab-gate`)

```bash
pip install weavetab-gate
```

```python
from weavetab_gate import Gate

gate = Gate()
res = gate.decide("refund requested after 45 days", ["approve", "escalate", "deny"])
print(res["choice"], res["confidence"])

with open("receipt.png", "rb") as f:
    img = gate.decide(f.read(), ["valid receipt", "blurry image", "spam"])
print(img["choice"], img["confidence"])
```

## Rust (`weavetab-gate`)

```rust
use weavetab_gate::{Gate, Input};

let gate = Gate::default();
let res = gate.decide(Input::Text("service unreachable"), &["outage", "question"])?;
println!("{} ({:.3})", res.choice, res.confidence);
```
