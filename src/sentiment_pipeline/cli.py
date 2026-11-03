"""Command-line entry point: parse args, wire components, map outcomes to exit codes."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .csv_io import InputSchemaError
from .metrics import Metrics, start_metrics_server
from .model import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION, HuggingFaceSentimentModel
from .output import OutputLocked, ResumeError, atomic_write_json
from .pipeline import PipelineAborted, PipelineConfig, RunInterrupted, run

log = logging.getLogger("sentiment_pipeline")

EXIT_OK = 0
EXIT_FATAL = 1  # nothing published: bad input, output locked, model load failed, run aborted
EXIT_FAILURE_RATE = 2  # output published, but too many rows failed
EXIT_INTERRUPTED = 130  # SIGINT / SIGTERM; progress is checkpointed, rerun to resume

# Errors that mean "this run cannot proceed as configured"; logged without a traceback.
FATAL_ERRORS = (InputSchemaError, OutputLocked, ResumeError, PipelineAborted)


class JsonFormatter(logging.Formatter):
    """One JSON object per line; structured fields come from `extra={"fields": {...}}`."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", None),
            "msg": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        fields = getattr(record, "fields", None)
        return f"{line} {json.dumps(fields, default=str)}" if fields else line


class RunIdFilter(logging.Filter):
    """Stamp every record with the run id so logs from one run can be correlated."""

    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def configure_logging(fmt: str, level: str, run_id: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RunIdFilter(run_id))
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Keep third-party chatter out of the structured log stream.
    for noisy in ("transformers", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _fraction(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sentiment_pipeline",
        description="Run batch sentiment inference over a CSV file.",
    )
    parser.add_argument("--input", required=True, type=Path, help="input CSV path")
    parser.add_argument("--output", type=Path, default=Path("output/predictions.csv"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", help="column to carry through as a stable row id")
    parser.add_argument("--label-column", help="ground-truth column; enables accuracy metric")
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument(
        "--window-size",
        type=_positive_int,
        default=1024,
        help="rows sorted by length before batching; also the checkpoint interval",
    )
    parser.add_argument("--max-length", type=_positive_int, default=512, help="max tokens per text")
    parser.add_argument(
        "--max-failure-rate",
        type=_fraction,
        default=0.05,
        help="exit with code 2 if a larger fraction of rows fails (default: 0.05)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore any checkpoint for this output and start from the first row",
    )
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument(
        "--model-revision", default=os.environ.get("MODEL_REVISION", DEFAULT_MODEL_REVISION)
    )
    parser.add_argument("--encoding", default="utf-8-sig", help="input file encoding")
    parser.add_argument(
        "--metrics-port", type=int, default=0, help="serve Prometheus /metrics on this port (0=off)"
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help="where to write the run summary JSON (default: <output>.metrics.json)",
    )
    parser.add_argument("--log-format", choices=["json", "text"], default="json")
    parser.add_argument("--log-level", default="INFO", type=str.upper)
    return parser.parse_args(argv)


def load_model(args: argparse.Namespace) -> HuggingFaceSentimentModel:
    start = time.perf_counter()
    model = HuggingFaceSentimentModel(args.model_name, args.model_revision, args.max_length)
    log.info(
        "model loaded",
        extra={
            "fields": {
                "model": args.model_name,
                "revision": args.model_revision,
                "seconds": round(time.perf_counter() - start, 2),
            }
        },
    )
    return model


def install_signal_handlers(stop: threading.Event) -> dict:
    """First SIGINT/SIGTERM: finish the current window, commit, exit 130.
    Second: stop immediately (still safe, the last commit is on disk).

    `docker stop` sends SIGTERM and follows with SIGKILL after 10s; a window
    normally takes a few seconds, so the graceful path usually wins.
    """

    def handle(signum: int, frame: object) -> None:
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        log.warning(
            "stop requested; finishing current window", extra={"fields": {"signal": signum}}
        )

    return {sig: signal.signal(sig, handle) for sig in (signal.SIGINT, signal.SIGTERM)}


def _package_version() -> str:
    try:
        return version("sentiment-pipeline")
    except PackageNotFoundError:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = uuid.uuid4().hex[:12]
    configure_logging(args.log_format, args.log_level, run_id)
    stop = threading.Event()
    previous_handlers = install_signal_handlers(stop)

    config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        text_column=args.text_column,
        id_column=args.id_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
        window_size=args.window_size,
        encoding=args.encoding,
        model_version=f"{args.model_name}@{args.model_revision}:max_length={args.max_length}",
        resume=not args.fresh,
    )
    metrics_file = args.metrics_file or args.output.with_suffix(".metrics.json")
    metrics = Metrics()
    metrics.info.update(
        run_id=run_id,
        version=_package_version(),
        model=args.model_name,
        model_revision=args.model_revision,
        input=str(args.input),
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    if args.metrics_port:
        start_metrics_server(metrics, args.metrics_port)
        log.info("metrics server started", extra={"fields": {"port": args.metrics_port}})

    status, exit_code = "failed", EXIT_FATAL
    try:
        run(config, lambda: load_model(args), metrics, stop)
        status, exit_code = "completed", EXIT_OK
    except FATAL_ERRORS as exc:
        log.error(str(exc))
    except (RunInterrupted, KeyboardInterrupt) as exc:
        status, exit_code = "interrupted", EXIT_INTERRUPTED
        log.warning(str(exc) or "run interrupted; rerun the same command to resume")
    except Exception:
        log.exception("run failed with an unexpected error")
    finally:
        summary = metrics.summary()
        if status == "completed" and summary["failure_rate"] > args.max_failure_rate:
            status, exit_code = "completed_with_errors", EXIT_FAILURE_RATE
            log.error(
                "failure rate above threshold",
                extra={
                    "fields": {
                        "failure_rate": summary["failure_rate"],
                        "max_failure_rate": args.max_failure_rate,
                    }
                },
            )
        report = {"status": status, "run": dict(metrics.info), **summary}
        atomic_write_json(metrics_file, report)
        log.info("run finished", extra={"fields": {"status": status, **summary}})
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return exit_code
