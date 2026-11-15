---
title: "Error Handling"
product: gate
section: gate-api
slug: gate-errors
description: "Structured JSON error envelope and status codes returned by api.weavetab.dev."
icon: AlertTriangle
order: 2
---

# Error Handling

Every non-`200` JSON response from `api.weavetab.dev` uses a predictable envelope:

```json
{
  "ok": false,
  "error": {
    "code": "INVALID_OPTIONS",
    "message": "Provide between 2 and 16 non-empty options.",
    "docs": "https://weavetab.dev/docs/gate/gate-errors"
  }
}
```

## Error Codes

| HTTP | `error.code` | Cause | Fix |
| :--- | :--- | :--- | :--- |
| `400` | `MISSING_USER_AGENT` | `User-Agent` header is missing or empty. | Send a `User-Agent` header (e.g. `my-app/1.0`). |
| `400` | `EMPTY_PROMPT` | `prompt` (or `q`) is empty. | Pass a non-empty string. |
| `400` | `INPUT_TOO_LONG` | `prompt` exceeds `1024` characters. | Trim or summarize input before calling. |
| `400` | `INVALID_OPTIONS` | Fewer than `2` or more than `16` options. | Pass 2–16 non-empty choice strings. |
| `413` | `PAYLOAD_TOO_LARGE` | Request body exceeds `4096` bytes (4 KB). | Keep payload under 4 KB. |
| `429` | `RATE_LIMITED` | Exceeded per-IP rate limit (180 req / 60s). | Back off or use `GET /v1/decide` for cacheable queries. |
| `503` | `SERVER_BUSY` | Concurrency queue (`64` in-flight) is full. | Retry after brief exponential backoff (250ms–1s). |
| `504` | `INFERENCE_TIMEOUT` | Inference exceeded `3s` hard limit. | Retry with a shorter prompt or fewer options. |
