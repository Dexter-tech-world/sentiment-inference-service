# syntax=docker/dockerfile:1
# Base pinned by digest (multi-arch index: amd64 + arm64) so rebuilds are reproducible.
# Bump deliberately, e.g. via Dependabot/Renovate.
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# 1. Dependencies: every package (including transitive ones) pinned and hash-verified.
#    --torch-backend cpu picks the CPU-only torch wheel (~2GB smaller than the CUDA build).
#    uv is mounted for this step only, so it never ends up in an image layer.
COPY requirements.lock .
RUN --mount=from=ghcr.io/astral-sh/uv:0.13.1,source=/uv,target=/bin/uv \
    uv pip install --system --no-cache --require-hashes --torch-backend cpu -r requirements.lock

# 2. Model weights, baked in so containers need no network and no Hub at runtime.
#    Kept above the source COPY so code changes don't re-download ~260MB.
ARG MODEL_NAME=distilbert/distilbert-base-uncased-finetuned-sst-2-english
ARG MODEL_REVISION=714eb0fa89d2f80546fda750413ed43d93601a13
ENV MODEL_NAME=${MODEL_NAME} MODEL_REVISION=${MODEL_REVISION}
COPY scripts/download_model.py /tmp/download_model.py
RUN python /tmp/download_model.py && chmod -R a+rX ${HF_HOME}

# 3. Application code.
COPY pyproject.toml requirements.in README.md ./
COPY src ./src
RUN --mount=from=ghcr.io/astral-sh/uv:0.13.1,source=/uv,target=/bin/uv \
    uv pip install --system --no-cache --no-deps .

# API_HOST: the API binds to all interfaces inside the container (default is 127.0.0.1).
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 API_HOST=0.0.0.0
RUN useradd --create-home --uid 1000 app && mkdir -p /data /output && chown app /output
USER app

# 8000: batch job /metrics (--metrics-port). 8080: HTTP API (--entrypoint sentiment-api).
EXPOSE 8000 8080
# Exec form: python is PID 1 and receives SIGTERM from `docker stop` directly.
ENTRYPOINT ["python", "-m", "sentiment_pipeline"]
CMD ["--help"]
