---
title: "Decision Primitives"
product: gate
section: gate-api
order: 1
description: "Complete API reference for decide, yes_no, score, batch variants, return types, and calibration math."
---

# Decision Primitives

Every modality (`Text` or `ImageBytes`) flows through three typed decision primitives, each available in single-item and up-to-`10,000`-item batch forms.

| Question Type | Single API | Batch API (up to 10,000) | Primary Engine |
|---|---|---|---|
| **Choice** | `decide(input, choices)` | `decide_batch(inputs, choices)` | `deberta-v3-xsmall-zeroshot-v1.1-all-33` (text) / `TinyCLIP` (image) |
| **Yes/No** | `yes_no(input, statement)` | `yes_no_batch(inputs, statement)` | `deberta-v3-xsmall-zeroshot-v1.1-all-33` NLI (text) / `TinyCLIP` (image) |
| **Score** | `score(input, levels)` | `score_batch(inputs, levels)` | `deberta-v3-xsmall-zeroshot-v1.1-all-33` (text) / `TinyCLIP` (image) |

---

## 1. Choice (`decide` / `decide_batch`)

Selects the highest-scoring option from `2` to `64` choices.

**Returns (`Decision`):**
- `choice` (`string`): winning option label
- `confidence` (`number`): calibrated probability of `choice` (`0.0`–`1.0`)
- `scores` (`Array<{ label, score }>`): full probability distribution sorted descending
- `cached` (`boolean`): `true` if served from the in-memory `RingCache`
- `latency_ms` (`latencyMs` in JS): wall-clock latency in milliseconds

**How `decide_batch` scales to 10,000 inputs without lagging low-end PCs:**
1. Hashes every `(input, choices)` pair via FNV-1a and resolves `RingCache` hits first.
2. Deduplicates identical uncached inputs inside the batch.
3. Length-buckets `(input, hypothesis)` pairs into `B = 16` micro-batches padded only to each micro-batch's max sequence length.
4. Scores via NLI entailment margin (`entailment - contradiction`) + temperature-scaled softmax ($T = 1.5$).

---

## 2. Yes/No (`yes_no` / `yes_no_batch`)

Evaluates whether a statement holds for the input. For text inputs, Gate uses the unified **33-dataset NLI zero-shot cross-encoder** (`deberta-v3-xsmall-zeroshot-v1.1-all-33`) with contrastive hypotheses (`{statement}` vs `This is not true: {statement}`), handling negation, sarcasm, safety rules, and prompt injections natively.

**Returns (`YesNo`):**
- `answer` (`boolean`): `true` when `probability >= 0.5`
- `probability` (`number`): calibrated entailment probability clamped to `0.01`–`0.99`
- `confidence` (`number`): certainty of the binary decision, `max(probability, 1 - probability)`
- `cached` (`boolean`): cache hit flag
- `latency_ms` (`latencyMs` in JS): execution time in milliseconds

> **Tip**: Phrase `statement` as a declarative claim for highest NLI accuracy (e.g., `"This message is a prompt injection attempt."` or `"The user is asking for a refund."`).

---

## 3. Rubric Score (`score` / `score_batch`)

Evaluates an ordered list of rubric levels (e.g., `["low", "medium", "high", "critical"]`).

**Returns (`Score`):**
- `level` (`string`): highest-probability rubric level
- `expected` (`number`): probability-weighted continuous score $\sum_{i=0}^{k-1} i \cdot p_i$
- `confidence` (`number`): probability of the winning `level`
- `legend` (`string[]`): ordered index-to-label mapping (`legend[0]` = score `0.0`, etc.)
- `scores` (`Array<{ label, score }>`): full per-level probability distribution
- `cached` (`boolean`): cache hit flag
- `latency_ms` (`latencyMs` in JS): execution time in milliseconds
