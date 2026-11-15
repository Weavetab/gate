---
title: "Decision Primitives"
product: gate
section: gate-api
order: 1
description: "API reference for decide, yes_no, score, and the 4096-entry RingCache."
---

# Decision Primitives

Every modality (`Text` or `ImageBytes`) flows through three structured primitives:

## `decide(input, choices)`
Evaluates between `2` and `64` candidate labels against a single input and returns:
- `choice`: highest-scoring label
- `confidence`: calibrated probability (`0.0` – `1.0`)
- `scores`: full sorted list of `{ label, score }`
- `cached`: `true` if served from the `4096`-entry RAM `RingCache`
- `latency_ms` (`latencyMs` in JS): wall-clock execution time in milliseconds

### Calibration Math
- **Text (`GLiClass`):** Applies per-label sigmoid $\sigma(x) = \frac{1}{1 + e^{-x}}$ followed by L1 normalization across the candidate set.
- **Vision (`TinyCLIP`):** Applies softmax $\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$ over `logits_per_image`.

## `yes_no(input, question)` / `yesNo(input, question)`
Evaluates a binary condition (`"yes: {question}"` vs `"no: {question}"`) and returns `{ answer: boolean, probability, cached, latency_ms }`.

## `score(input, levels)`
Evaluates an ordered rubric (e.g. `["low", "medium", "high", "critical"]`) and returns `{ level, expected, scores, cached, latency_ms }`, where `expected = sum(i * p_i)`.
