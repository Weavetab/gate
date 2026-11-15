FROM rust:1.83-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/weavetab-gate /usr/local/bin/weavetab-gate
RUN mkdir -p /app/model \
  && curl -fsSL -o /app/model/model_i8.onnx https://huggingface.co/cnmoro/gliclass-edge-v3.0-onnx/resolve/main/model_i8.onnx \
  && curl -fsSL -o /app/model/tokenizer.json https://huggingface.co/cnmoro/gliclass-edge-v3.0-onnx/resolve/main/tokenizer.json
ENV PORT=7860
EXPOSE 7860
CMD ["weavetab-gate"]
