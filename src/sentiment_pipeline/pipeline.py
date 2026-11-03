"""Batch inference loop.

    stream rows -> window of N rows -> validate -> sort by length -> predict in
    batches -> restore input order -> append to output -> commit checkpoint

Invariant: every input record produces exactly one output record, with
status "ok" or "failed", so rows_read == rows_succeeded + rows_failed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import batched, islice
from pathlib import Path

from .csv_io import InputRow, read_rows
from .metrics import EMPTY_TEXT, INFERENCE_ERROR, MALFORMED_CSV, Metrics
from .model import Prediction, SentimentModel
from .output import CheckpointedCsvWriter, output_lock

log = logging.getLogger(__name__)


class PipelineAborted(RuntimeError):
    """Inference is failing systematically; continuing would only burn time."""


class RunInterrupted(Exception):
    """A stop was requested. Everything up to the last window is committed."""


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_path: Path
    text_column: str = "text"
    id_column: str | None = None
    label_column: str | None = None
    batch_size: int = 32
    # Rows sorted together by length before batching, and the checkpoint interval.
    window_size: int = 1024
    encoding: str = "utf-8-sig"
    model_version: str = "unknown"  # part of the resume fingerprint
    resume: bool = True
    # Abort if this many batches in a row have every inference attempt fail
    # (e.g. a broken model): row-by-row fallback would be ~batch_size x slower.
    max_consecutive_failed_batches: int = 3
    log_every_windows: int = 10

    def output_fieldnames(self) -> list[str]:
        fields = ["row_number"]
        if self.id_column:
            fields.append("id")
        fields.append("text")
        if self.label_column:
            fields.append("label")
        return fields + ["predicted_label", "score", "status", "error"]

    def fingerprint(self, input_sha256: str) -> str:
        """Everything that determines the output's content. Resume requires an exact match.

        Batch and window size are deliberately left out: they change speed, not results.
        """
        identity = {
            "input_sha256": input_sha256,
            "text_column": self.text_column,
            "id_column": self.id_column,
            "label_column": self.label_column,
            "encoding": self.encoding,
            "model_version": self.model_version,
            "fields": self.output_fieldnames(),
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def run(
    config: PipelineConfig,
    load_model: Callable[[], SentimentModel],
    metrics: Metrics,
    stop: threading.Event | None = None,
) -> None:
    # Cheap checks first: a bad header, or another run holding the output, fails
    # before we pay for hashing the input or loading the model.
    rows = read_rows(
        config.input_path,
        text_column=config.text_column,
        id_column=config.id_column,
        label_column=config.label_column,
        encoding=config.encoding,
    )
    with output_lock(config.output_path):
        input_sha256 = sha256_file(config.input_path)
        metrics.info["input_sha256"] = input_sha256
        writer = CheckpointedCsvWriter(
            config.output_path, config.output_fieldnames(), config.fingerprint(input_sha256)
        )
        checkpoint = writer.load_checkpoint() if config.resume else None
        rows_done = 0
        if checkpoint is not None:
            metrics.restore(checkpoint.metrics)
            rows_done = checkpoint.rows_committed
            rows = islice(rows, rows_done, None)  # re-reading is cheap next to inference
            log.info("resuming from checkpoint", extra={"fields": {"rows_committed": rows_done}})

        model = load_model()
        metrics.started_at = time.monotonic()  # throughput excludes model load time
        breaker = _CircuitBreaker(config.max_consecutive_failed_batches)
        log.info(
            "run started",
            extra={"fields": {"input": str(config.input_path), "batch_size": config.batch_size}},
        )

        with writer.open(checkpoint):
            for window_number, window in enumerate(batched(rows, config.window_size), start=1):
                writer.write_rows(_process_window(window, model, metrics, config, breaker))
                rows_done += len(window)
                writer.commit(rows_done, metrics.state())

                if window_number % config.log_every_windows == 0:
                    log.info("progress", extra={"fields": metrics.summary()})
                if stop is not None and stop.is_set():
                    raise RunInterrupted(
                        f"stopped after {rows_done} rows; rerun the same command to resume"
                    )
            writer.publish()

    log.info("output written", extra={"fields": {"output": str(config.output_path)}})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class _CircuitBreaker:
    """Raise PipelineAborted after `limit` consecutive batches where every row failed."""

    def __init__(self, limit: int):
        self.limit = limit
        self.consecutive = 0

    def record(self, attempted: int, failed: int) -> None:
        self.consecutive = self.consecutive + 1 if failed == attempted else 0
        if self.consecutive >= self.limit:
            raise PipelineAborted(
                f"aborting: every row failed inference in {self.consecutive} consecutive batches"
            )


def _process_window(
    window: Sequence[InputRow],
    model: SentimentModel,
    metrics: Metrics,
    config: PipelineConfig,
    breaker: _CircuitBreaker,
) -> list[dict]:
    """Return one output record per input row, in input order."""
    records = [_base_record(row, config) for row in window]

    pending: list[int] = []  # indices of rows that pass validation
    for i, row in enumerate(window):
        if row.error:
            _fail(records[i], MALFORMED_CSV, row.error, metrics)
        elif row.text is None or not row.text.strip():
            _fail(records[i], EMPTY_TEXT, "text is empty", metrics)
        else:
            pending.append(i)

    # Every batch is padded to its longest text, so batching similar lengths
    # together cuts wasted compute (measured 1.4-1.8x faster). Results are
    # written back by index, so output order is unaffected.
    pending.sort(key=lambda i: len(window[i].text))

    for chunk in batched(pending, config.batch_size):
        outcomes = _predict(model, [window[i].text for i in chunk], metrics)
        failed = 0
        for i, outcome in zip(chunk, outcomes, strict=True):
            if isinstance(outcome, Exception):
                failed += 1
                _fail(records[i], INFERENCE_ERROR, f"{type(outcome).__name__}: {outcome}", metrics)
                continue
            records[i].update(predicted_label=outcome.label, score=outcome.score, status="ok")
            label = window[i].label
            expected = label.strip().lower() if label and label.strip() else None
            metrics.record_prediction(outcome.label, outcome.score, expected)
        breaker.record(attempted=len(chunk), failed=failed)
    return records


def _predict(
    model: SentimentModel, texts: list[str], metrics: Metrics
) -> list[Prediction | Exception]:
    """Predict the whole batch; if that fails, retry row by row to isolate bad inputs.

    One poison row then costs one failed row instead of a whole failed batch.
    """
    start = time.perf_counter()
    try:
        predictions = model.predict(texts)
        if len(predictions) != len(texts):
            raise RuntimeError(
                f"model returned {len(predictions)} predictions for {len(texts)} inputs"
            )
        metrics.record_batch(time.perf_counter() - start, fallback=False)
        return predictions
    except Exception:
        log.warning(
            "batch inference failed; retrying rows individually",
            exc_info=True,
            extra={"fields": {"batch_rows": len(texts)}},
        )

    outcomes: list[Prediction | Exception] = []
    for text in texts:
        try:
            (prediction,) = model.predict([text])  # unpacking also enforces exactly one result
            outcomes.append(prediction)
        except Exception as exc:
            outcomes.append(exc)
    metrics.record_batch(time.perf_counter() - start, fallback=True)
    return outcomes


def _base_record(row: InputRow, config: PipelineConfig) -> dict:
    record = {
        "row_number": row.row_number,
        "text": row.text,
        "predicted_label": None,
        "score": None,
        "status": None,
        "error": None,
    }
    if config.id_column:
        record["id"] = row.id
    if config.label_column:
        record["label"] = row.label
    return record


def _fail(record: dict, reason: str, detail: str, metrics: Metrics) -> None:
    record.update(status="failed", error=f"{reason}: {detail}")
    metrics.record_failure(reason)
    level = logging.DEBUG if reason == EMPTY_TEXT else logging.WARNING
    log.log(
        level,
        "row failed",
        extra={"fields": {"row_number": record["row_number"], "reason": reason, "error": detail}},
    )
