---
title: "Introduction"
product: gate
section: gate-get-started
slug: gate-introduction
description: "Open, keyless, single-pass decision API at api.weavetab.dev — built in Rust on INT8 ONNX for real-time routing, bots, and classification."
icon: Zap
order: 1
---

# Weavetab Gate

**Weavetab Gate** (`api.weavetab.dev`) is an open, keyless decision engine powered by `weavetab-gate-1` — a quantized INT8 ONNX encoder served by an `AGPL-3.0` Rust (`axum` + `ort`) gateway.

Instead of spinning up a generative LLM to output text token-by-token just to make a routing or yes/no decision, Gate scores all of your custom options in a **single forward pass**.

## Why Use Gate?

* **Zero API Keys:** Call `https://api.weavetab.dev/v1/decide` directly from any language. No signup, no billing, no OAuth.
* **Single-Pass Speed:** No autoregressive token generation loop — ideal for real-time loops like Minecraft bots (2+ actions/sec), moderation filters, and agent routers.
* **Any Language, Any Choices:** Pass your own `options` array (`["yes", "no"]`, `["attack", "flee", "mine", "trade"]`, etc.) and receive calibrated confidence scores.
* **Stateless & Zero-Retention:** Prompts are evaluated in RAM and discarded immediately. Zero database, zero prompt logs.
