"""HTTP API for real-time predictions: a Flask app served by gunicorn.

Shares validation, the English check, fault-isolating inference and metrics
with the batch pipeline (see inference.py), so both entry points classify
identically.

Contract:
- POST /v1/predict  {"texts": ["...", ...]}  ->  one result per text, in order.
  Request-level problems (not JSON, wrong types, limits exceeded) return 4xx.
  Content problems (empty text, non-English, an inference error) are reported
  per item with status "failed" and never fail the whole request.
- GET /healthz  liveness: the process is serving.
- GET /readyz   readiness: the model is loaded.
- GET /metrics  Prometheus exposition.

    sentiment-api --port 8080            # or: python -m sentiment_pipeline.api
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading
import time
import uuid

from flask import Flask, Response, g, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from werkzeug.exceptions import BadRequest, HTTPException, UnsupportedMediaType

from .cli import configure_logging
from .inference import predict_with_fallback, validate_text
from .language import default_detector
from .metrics import INFERENCE_ERROR, Metrics, build_registry
from .model import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    HuggingFaceSentimentModel,
    SentimentModel,
)

log = logging.getLogger("sentiment_pipeline.api")

MAX_TEXTS_PER_REQUEST = 64
MAX_TEXT_CHARS = 10_000
MAX_CONTENT_LENGTH = 1 << 20  # 1 MiB; larger bodies get 413 before being parsed
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def create_app(
    model: SentimentModel | None = None,
    *,
    model_version: str | None = None,
    language_check: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
    model_revision: str = DEFAULT_MODEL_REVISION,
    max_length: int = 512,
) -> Flask:
    """Build the app. Pass `model` to inject one (tests); otherwise the pinned model is loaded."""
    if model is None:
        start = time.perf_counter()
        model = HuggingFaceSentimentModel(model_name, model_revision, max_length)
        model_version = f"{model_name}@{model_revision}"
        log.info(
            "model loaded",
            extra={
                "fields": {"model": model_version, "seconds": round(time.perf_counter() - start, 2)}
            },
        )
    model_version = model_version or "unknown"
    detector = default_detector() if language_check else None

    # PyTorch already parallelises one forward pass across all cores, so
    # concurrent passes in one process only oversubscribe the CPU. Serialise
    # inference per process and scale out with gunicorn workers instead.
    inference_lock = threading.Lock()

    metrics = Metrics(info={"instance_id": uuid.uuid4().hex[:12], "model": model_version})
    registry = build_registry(metrics)
    http_requests = Counter(
        "sentiment_http_requests",
        "HTTP requests by endpoint and status",
        ["method", "endpoint", "status"],
        registry=registry,
    )
    http_latency = Histogram(
        "sentiment_http_request_duration_seconds",
        "HTTP request latency",
        ["endpoint"],
        registry=registry,
    )

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.before_request
    def _start() -> None:
        g.start = time.perf_counter()
        incoming = request.headers.get("X-Request-ID", "")
        g.request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex[:16]

    @app.after_request
    def _finish(response: Response) -> Response:
        # Route templates, not raw paths, so label cardinality stays bounded.
        endpoint = request.url_rule.rule if request.url_rule else "unmatched"
        elapsed = time.perf_counter() - g.start
        http_requests.labels(request.method, endpoint, str(response.status_code)).inc()
        http_latency.labels(endpoint).observe(elapsed)
        response.headers["X-Request-ID"] = g.request_id
        if endpoint not in ("/healthz", "/readyz", "/metrics"):
            log.info(
                "request",
                extra={
                    "fields": {
                        "request_id": g.request_id,
                        "method": request.method,
                        "path": request.path,
                        "status": response.status_code,
                        "ms": round(elapsed * 1000, 1),
                    }
                },
            )
        return response

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        body = {"error": {"code": exc.code, "name": exc.name, "message": exc.description}}
        return jsonify(body), exc.code

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception):
        log.exception("unhandled error", extra={"fields": {"request_id": g.get("request_id")}})
        body = {
            "error": {"code": 500, "name": "Internal Server Error", "message": "unexpected error"}
        }
        return jsonify(body), 500

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    @app.get("/readyz")
    def readyz():
        return jsonify(status="ready", model=model_version, language_check=language_check)

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(generate_latest(registry), content_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/predict")
    def predict():
        texts = _parse_texts()
        results: list[dict | None] = [None] * len(texts)

        pending: list[int] = []
        for i, text in enumerate(texts):
            if rejection := validate_text(text, detector):
                results[i] = _failed(f"{rejection.reason}: {rejection.detail}")
                metrics.record_failure(rejection.reason)
            else:
                pending.append(i)

        if pending:
            with inference_lock:
                outcomes = predict_with_fallback(model, [texts[i] for i in pending], metrics)
            for i, outcome in zip(pending, outcomes, strict=True):
                if isinstance(outcome, Exception):
                    results[i] = _failed(f"{INFERENCE_ERROR}: {type(outcome).__name__}")
                    metrics.record_failure(INFERENCE_ERROR)
                    continue
                results[i] = {
                    "label": outcome.label,
                    "score": outcome.score,
                    "status": "ok",
                    "error": None,
                }
                metrics.record_prediction(outcome.label, outcome.score)

        return jsonify(model=model_version, predictions=results)

    return app


def _parse_texts() -> list[str]:
    if not request.is_json:
        raise UnsupportedMediaType("Content-Type must be application/json")
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise BadRequest("request body must be a JSON object")
    texts = payload.get("texts")
    if not isinstance(texts, list) or not texts:
        raise BadRequest('"texts" must be a non-empty array of strings')
    if len(texts) > MAX_TEXTS_PER_REQUEST:
        raise BadRequest(f'"texts" may contain at most {MAX_TEXTS_PER_REQUEST} items')
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise BadRequest(f'"texts[{i}]" must be a string')
        if len(text) > MAX_TEXT_CHARS:
            raise BadRequest(f'"texts[{i}]" exceeds {MAX_TEXT_CHARS} characters')
    return texts


def _failed(error: str) -> dict:
    # Inference errors expose only the exception type: messages can echo input or internals.
    return {"label": None, "score": None, "status": "failed", "error": error}


def main(argv: list[str] | None = None) -> None:
    from gunicorn.app.base import BaseApplication

    parser = argparse.ArgumentParser(
        prog="sentiment-api", description="Serve real-time sentiment predictions over HTTP."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("API_HOST", "127.0.0.1"),
        help="bind address (default 127.0.0.1; the Docker image sets 0.0.0.0)",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("API_WORKERS", "1")),
        help="processes; each loads its own model (~1 GB)",
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="seconds before a stuck worker is restarted"
    )
    parser.add_argument("--no-language-check", action="store_true")
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument(
        "--model-revision", default=os.environ.get("MODEL_REVISION", DEFAULT_MODEL_REVISION)
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--log-format", choices=["json", "text"], default="json")
    parser.add_argument("--log-level", default="INFO", type=str.upper)
    args = parser.parse_args(argv)
    configure_logging(args.log_format, args.log_level, run_id=uuid.uuid4().hex[:12])

    class Server(BaseApplication):
        def load_config(self) -> None:
            options = {
                "bind": f"{args.host}:{args.port}",
                "workers": args.workers,
                "timeout": args.timeout,
                "graceful_timeout": 30,
                # The model is loaded in each worker after fork, never in the
                # master: forking after PyTorch has started its thread pools can deadlock.
                "preload_app": False,
                "accesslog": None,  # requests are logged as structured JSON by the app
                "loglevel": args.log_level.lower(),
            }
            if os.path.isdir("/dev/shm"):
                # Worker heartbeats on a RAM disk: works with a read-only root
                # filesystem and avoids stalls on slow container overlays.
                options["worker_tmp_dir"] = "/dev/shm"
            for key, value in options.items():
                self.cfg.set(key, value)

        def load(self) -> Flask:
            return create_app(
                language_check=not args.no_language_check,
                model_name=args.model_name,
                model_revision=args.model_revision,
                max_length=args.max_length,
            )

    Server().run()


if __name__ == "__main__":
    main()
