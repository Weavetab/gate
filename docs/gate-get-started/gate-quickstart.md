---
title: "Quickstart"
product: gate
section: gate-get-started
slug: gate-quickstart
description: "Call api.weavetab.dev/v1/decide in 10 seconds from curl, JavaScript, Python, Go, or Rust with zero packages."
icon: Terminal
order: 2
---

# Quickstart

No SDK or API key is required. Include a descriptive `User-Agent` header and send a `POST` or `GET` request to `https://api.weavetab.dev/v1/decide`.

## cURL

```bash
curl -X POST https://api.weavetab.dev/v1/decide \
  -H "Content-Type: application/json" \
  -H "User-Agent: my-app/1.0" \
  -d '{
    "prompt": "Zombie is 3 blocks away and player health is 4/20",
    "options": ["fight", "flee", "eat_food", "build_wall"]
  }'
```

## JavaScript / TypeScript (`fetch`)

```ts
const res = await fetch("https://api.weavetab.dev/v1/decide", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "User-Agent": "mc-bot/1.0",
  },
  body: JSON.stringify({
    prompt: "Player asked: can I have OP?",
    options: ["yes", "no"],
  }),
});

const data = await res.json();
// { ok: true, model: "weavetab-gate-1", decision: "no", confidence: 0.94, ... }
```

## Python (`requests`)

```python
import requests

r = requests.post(
    "https://api.weavetab.dev/v1/decide",
    headers={"User-Agent": "my-service/1.0"},
    json={
        "prompt": "Cancel my subscription and refund me",
        "options": ["billing_refund", "tech_support", "spam"],
    },
    timeout=5,
)
print(r.json()["decision"])
```

## Edge-Cached `GET` (Fastest for Repeated Prompts)

```bash
curl -H "User-Agent: my-app/1.0" \
  "https://api.weavetab.dev/v1/decide?q=Server+CPU+is+at+99%25&options=scale_up,ignore,reboot"
```
