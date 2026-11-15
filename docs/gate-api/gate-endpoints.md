---
title: "Endpoints & Limits"
product: gate
section: gate-api
slug: gate-endpoints
description: "API reference for POST /v1/decide, GET /v1/decide, GET /health, rate limits, and payload constraints."
icon: Globe
order: 1
---

# Endpoints & Limits

Base URL: `https://api.weavetab.dev`

## `POST /v1/decide`

Evaluates `prompt` against `options` in a single forward pass.

### Request Headers
| Header | Required | Notes |
| :--- | :--- | :--- |
| `Content-Type` | Yes | `application/json` |
| `User-Agent` | Yes | Any non-empty identifier (e.g. `mc-bot/1.2`). Requests without a `User-Agent` return `400`. |

### Request Body
```json
{
  "prompt": "string (1..1024 chars)",
  "options": ["yes", "no"]
}
```
* `options` is optional — defaults to `["yes", "no"]`. Provide between **2 and 16** non-empty strings.
* Max request body size is **4 KB (4096 bytes)**.

### Response (`200 OK`)
```json
{
  "ok": true,
  "model": "weavetab-gate-1",
  "decision": "yes",
  "confidence": 0.9182,
  "scores": {
    "yes": 0.9182,
    "no": 0.0818
  },
  "cached": false
}
```

## `GET /v1/decide`

Same output as `POST /v1/decide`, with `Cache-Control: public, max-age=3600` so Cloudflare caches identical queries at the edge.

```text
GET /v1/decide?q=<url-encoded-prompt>&options=opt1,opt2
```

## `GET /health`

Returns `{"ok":true,"model":"weavetab-gate-1","status":"ready"}`.

## Rate Limits
* **180 requests / 60 seconds per IP** (enforced at the Cloudflare edge; supports sustained 2–3 actions/sec per bot).
* Exceeding the limit returns `429 Too Many Requests`.
