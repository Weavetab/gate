---
title: "Privacy & Licensing"
product: gate
section: gate-privacy
slug: gate-privacy
description: "Zero-retention stateless inference architecture, no prompt logging, AGPL-3.0 gateway, and Apache-2.0 model attribution."
icon: Shield
order: 1
---

# Privacy & Zero-Retention Architecture

Weavetab Gate is designed as a **stateless brain**: it scores your input in memory and forgets it the millisecond the HTTP response is sent.

## What We Do NOT Store
* **Zero Prompt Storage:** Prompts and `options` are never written to disk, databases, or log files.
* **Zero Training:** Your API traffic is never collected or used to train models.
* **Zero Tracking:** No cookies, no accounts, no API keys.
* **In-Memory Cache Only:** A bounded 4,096-entry RAM ring cache speeds up identical repeated queries and clears automatically on restart.

## Open Source & Licensing
You can audit the entire gateway source code in [`weavetab/gate`](https://github.com/weavetab/gate):
* **Gateway & Server (`src/`, `worker/`, `Dockerfile`):** Licensed under **GNU Affero General Public License v3.0 (`AGPL-3.0`)**. Anyone hosting a modified version over a network must publish their server changes.
* **Model Weights (`weavetab/gate-1`):** Licensed under **Apache License 2.0**, derived from `knowledgator/gliclass-edge-v3.0` (Knowledgator) and `cnmoro/gliclass-edge-v3.0-onnx` with full `LICENSE` and `NOTICE` attribution preserved.
