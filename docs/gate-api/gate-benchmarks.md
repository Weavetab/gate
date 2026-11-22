---
title: "Benchmarks"
product: gate
section: gate-api
order: 3
description: "Reproducible local CPU benchmarks for Weavetab Gate v1 vs cloud decision APIs."
---

# Benchmarks

All numbers below were measured on a standard Windows x64 CPU (release Rust build, `threads = 4`, zero GPU) against Google's `gemini-decision-1` cloud API over 34 realistic test cases (Support Routing, Safety/Prompt-Injection Yes/No, Severity Rubric, and Hard Negation/Sarcasm) plus a 10,000-item single-call batch test.

## Head-to-Head Summary

| Metric | **Weavetab Gate v1** (Local CPU) | **Google `gemini-decision-1`** (Cloud API) |
|---|---|---|
| **Routing Latency (12 items)** | **`10.9 ms/item`** (`131 ms` total) | `415.8 ms/item` (`4,990 ms` total) |
| **Yes/No Safety Accuracy (8 items)** | **`88%` (`7/8`)** via NLI engine | `88%` (`7/8`, **1 blocked**) |
| **Yes/No Latency (8 items)** | **`83.2 ms/item`** (`665 ms` total) | `258.7 ms/item` (`2,070 ms` total) |
| **Content-Policy Blocks** | **`0`** (runs 100% locally) | **`1`** (`400 prohibited_content`) |
| **10,000 Unique Inputs (1 Call)** | **`72.9 s`** (`137.2 items/s`, `7.29 ms/item`) | Requires multiple API calls + rate limits |
| **Cached Hit Latency** | **`0 ms`** (`RingCache`) | Network RTT on every call |
| **Cost** | **`$0`** (Apache-2.0, offline) | Rate-limited free tier / `$0.10` per 1M input tokens |

## Reproducing Locally

```bash
# Single-item text + vision + 5-case routing eval
cargo run --release --example bench

# 10,000 unique items in one call
cargo run --release --example bench -- --batch

# Full 34-case Python comparison suite
python scripts/bench_compare.py
```
