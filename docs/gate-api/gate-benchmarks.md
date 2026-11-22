---
title: "Benchmarks"
product: gate
section: gate-api
order: 3
description: "Reproducible local CPU & GPU benchmarks for Weavetab Gate v1.1 vs cloud decision APIs."
---

# Benchmarks

All numbers below were measured on a **Linux x64 4-vCPU CPU (16 GB RAM, no GPU)** and **Tesla T4 GPU** against Google's `gemini-decision-1` cloud API over 56 realistic test cases across 6 suites (Support Routing, Safety/Prompt-Injection Yes/No, Sentiment, Spam/Intent, Severity Rubric, and Hard Negation/Sarcasm).

## Head-to-Head Summary (`72.2 MB` Unified ONNX)

| Suite (Cases) | **Weavetab Gate v1.1** (4-vCPU Linux CPU) | **Gate v1.1** (Tesla T4 GPU) | **Google `gemini-decision-1`** (Cloud API) | Speedup (CPU vs Cloud) |
|---|---|---|---|---|
| **Overall (56 cases)** | **`92.9%` (`52/56`)** (+25.0 pts vs v1.0) | **`92.9%`** · `5.0–8.2 ms` steady-state | `92.9%` (`52/56`, **1 blocked**) | **3.7×–8.9× faster** |
| **Routing (12)** | **`100.0%` (`12/12`)** · `109.0 ms/item` | `78.6 ms/item` (cold init) | `100.0%` (`12/12`) · `701.5 ms/item` | **6.4× faster** |
| **Safety Yes/No (10)** | **`100.0%` (`10/10`)** · `65.0 ms/item` | `5.2 ms/item` | `90.0%` (`9/10`, **1 blocked**) · `389.0 ms/item` | **6.0× faster** (+10 pts) |
| **Sentiment (10)** | **`100.0%` (`10/10`)** · `67.5 ms/item` | `5.1 ms/item` | `100.0%` (`10/10`) · `333.1 ms/item` | **4.9× faster** |
| **Spam/Intent (10)** | **`90.0%` (`9/10`)** · `83.6 ms/item` | `5.0 ms/item` | `90.0%` (`9/10`) · `313.3 ms/item` | **3.7× faster** |
| **Severity Rubric (8)** | `62.5%` (`5/8`) · `68.6 ms/item` | `6.1 ms/item` | `75.0%` (`6/8`) · `375.1 ms/item` | **5.5× faster** |
| **Hard Negation/Sarcasm (6)** | **`100.0%` (`6/6`)** · `54.2 ms/item` | `8.2 ms/item` | `100.0%` (`6/6`) · `480.4 ms/item` | **8.9× faster** |
| **Content-Policy Blocks** | **`0`** (runs 100% locally) | **`0`** | **`1`** (`400 prohibited_content`) | — |
| **Cost & Footprint** | **`$0`** · `72.2 MB` unified ONNX | **`$0`** | Rate-limited / `$0.10` per 1M input tokens | — |

## Visual Comparisons

![Chart 1: Zero-Shot Accuracy Across 6 Decision Suites](/images/gate/chart1_gate_accuracy.png?v=1.1)

![Chart 2: Per-Decision Latency Comparison](/images/gate/chart2_gate_latency.png?v=1.1)

![Chart 3: Gate v1.1 Scaling & Head-to-Head Scorecard](/images/gate/chart3_gate_scaling.png?v=1.1)

## Reproducing Locally

```bash
# Single-item text + vision + 5-case routing eval
cargo run --release --example bench

# 10,000 unique items in one call
cargo run --release --example bench -- --batch

# Full comparison suite
python scripts/bench_compare.py
```
