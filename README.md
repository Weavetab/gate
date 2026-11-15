# Weavetab Gate (`api.weavetab.dev`)

Open, keyless, single-pass decision API built in Rust (`axum` + `gliclass-rs` / ONNX Runtime). Send a prompt + custom choices and get calibrated JSON decisions in one forward pass — zero API keys, zero prompt retention.

## Quick Example

```bash
curl -X POST https://api.weavetab.dev/v1/decide \
  -H "Content-Type: application/json" \
  -H "User-Agent: my-bot/1.0" \
  -d '{"prompt":"Player asked to trade 64 diamonds for 1 dirt","options":["accept","reject","counter"]}'
```

## License

* **Server & Gateway (`src/`, `worker/`, `Dockerfile`):** GNU Affero General Public License v3.0 (`AGPL-3.0`) — see [LICENSE](./LICENSE).
* **Model Weights (`weavetab/gate-1`):** Apache License 2.0, derived from `knowledgator/gliclass-edge-v3.0` and `cnmoro/gliclass-edge-v3.0-onnx` — see [NOTICE](./NOTICE).
