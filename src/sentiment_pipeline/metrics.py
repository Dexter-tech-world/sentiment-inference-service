"""Run metrics: one in-memory source of truth, exported as JSON and Prometheus.

The pipeline thread writes; the optional /metrics HTTP thread reads. A lock
keeps snapshots consistent (e.g. rows_read always equals succeeded + failed).
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, InfoMetricFamily

# Failure reasons are a fixed, small set so they are safe as Prometheus labels.
# Exception messages go to logs and the output file, never into label values.
EMPTY_TEXT = "empty_text"
NON_ENGLISH = "non_english"
MALFORMED_CSV = "malformed_csv"
INFERENCE_ERROR = "inference_error"
FAILURE_REASONS = (EMPTY_TEXT, NON_ENGLISH, MALFORMED_CSV, INFERENCE_ERROR)

# For a binary classifier the top score is always >= 0.5; below this it is
# close to a coin flip. A rising share of these is an early sign of drift.
LOW_CONFIDENCE_THRESHOLD = 0.7

# Cumulative counters, persisted in checkpoints so a resumed run reports totals
# for the whole output rather than for the last session only.
_PERSISTED = (
    "rows_succeeded",
    "rows_failed",
    "failures_by_reason",
    "predictions_by_label",
    "low_confidence_predictions",
    "batches_total",
    "batch_fallbacks",
    "inference_seconds",
    "labelled_rows",
    "correct_predictions",
)


@dataclass
class Metrics:
    rows_succeeded: int = 0
    rows_failed: int = 0
    failures_by_reason: Counter = field(default_factory=Counter)
    predictions_by_label: Counter = field(default_factory=Counter)
    low_confidence_predictions: int = 0
    batches_total: int = 0
    batch_fallbacks: int = 0  # batches that failed and were retried row by row
    inference_seconds: float = 0.0
    labelled_rows: int = 0
    correct_predictions: int = 0
    rows_resumed: int = 0  # rows carried over from a checkpoint, not processed this session
    info: dict[str, str] = field(default_factory=dict)  # static run metadata
    started_at: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_prediction(self, label: str, score: float, expected: str | None = None) -> None:
        with self._lock:
            self.rows_succeeded += 1
            self.predictions_by_label[label] += 1
            self.low_confidence_predictions += int(score < LOW_CONFIDENCE_THRESHOLD)
            if expected is not None:
                self.labelled_rows += 1
                self.correct_predictions += int(label == expected)

    def record_failure(self, reason: str) -> None:
        with self._lock:
            self.rows_failed += 1
            self.failures_by_reason[reason] += 1

    def record_batch(self, seconds: float, fallback: bool) -> None:
        with self._lock:
            self.batches_total += 1
            self.batch_fallbacks += int(fallback)
            self.inference_seconds += seconds

    def state(self) -> dict:
        """JSON-serialisable counters for a checkpoint."""
        state = {}
        with self._lock:
            for name in _PERSISTED:
                value = getattr(self, name)
                state[name] = dict(value) if isinstance(value, Counter) else value
        return state

    def restore(self, state: dict) -> None:
        """Continue counting from a checkpoint."""
        with self._lock:
            for name in _PERSISTED:
                current = getattr(self, name)
                setattr(
                    self,
                    name,
                    Counter(state[name]) if isinstance(current, Counter) else state[name],
                )
            self.rows_resumed = self.rows_succeeded + self.rows_failed

    def summary(self) -> dict:
        with self._lock:
            rows_read = self.rows_succeeded + self.rows_failed
            elapsed = time.monotonic() - self.started_at
            processed_now = rows_read - self.rows_resumed
            return {
                "rows_read": rows_read,
                "rows_succeeded": self.rows_succeeded,
                "rows_failed": self.rows_failed,
                "rows_resumed": self.rows_resumed,
                "failure_rate": round(self.rows_failed / rows_read, 4) if rows_read else 0.0,
                "failures_by_reason": dict(self.failures_by_reason),
                "predictions_by_label": dict(self.predictions_by_label),
                "low_confidence_predictions": self.low_confidence_predictions,
                "batches_total": self.batches_total,
                "batch_fallbacks": self.batch_fallbacks,
                "elapsed_seconds": round(elapsed, 3),
                "inference_seconds": round(self.inference_seconds, 3),
                "rows_per_second": round(processed_now / elapsed, 1) if elapsed > 0 else 0.0,
                "accuracy": (
                    round(self.correct_predictions / self.labelled_rows, 4)
                    if self.labelled_rows
                    else None
                ),
            }


class _PrometheusCollector:
    """Renders a Metrics snapshot in Prometheus format on every scrape."""

    def __init__(self, metrics: Metrics):
        self.metrics = metrics

    def collect(self):
        s = self.metrics.summary()

        yield InfoMetricFamily(
            "sentiment_run", "Static run metadata", value=dict(self.metrics.info)
        )

        rows = CounterMetricFamily("sentiment_rows", "Rows processed by outcome", labels=["status"])
        rows.add_metric(["succeeded"], s["rows_succeeded"])
        rows.add_metric(["failed"], s["rows_failed"])
        yield rows

        failures = CounterMetricFamily(
            "sentiment_row_failures", "Failed rows by reason", labels=["reason"]
        )
        for reason in FAILURE_REASONS:
            failures.add_metric([reason], s["failures_by_reason"].get(reason, 0))
        yield failures

        # Label values come from the model's fixed id2label map, so cardinality is bounded.
        predictions = CounterMetricFamily(
            "sentiment_predictions", "Successful predictions by label", labels=["label"]
        )
        for label, count in sorted(s["predictions_by_label"].items()):
            predictions.add_metric([label], count)
        yield predictions

        yield CounterMetricFamily(
            "sentiment_low_confidence_predictions",
            f"Predictions with score below {LOW_CONFIDENCE_THRESHOLD}",
            value=s["low_confidence_predictions"],
        )
        yield CounterMetricFamily("sentiment_batches", "Batches run", value=s["batches_total"])
        yield CounterMetricFamily(
            "sentiment_batch_fallbacks",
            "Batches that failed and were retried row by row",
            value=s["batch_fallbacks"],
        )
        yield CounterMetricFamily(
            "sentiment_inference_seconds",
            "Time spent in model inference",
            value=s["inference_seconds"],
        )
        yield GaugeMetricFamily(
            "sentiment_rows_per_second",
            "Throughput of this session so far",
            value=s["rows_per_second"],
        )


def build_registry(metrics: Metrics) -> CollectorRegistry:
    registry = CollectorRegistry()
    registry.register(_PrometheusCollector(metrics))
    return registry


def start_metrics_server(metrics: Metrics, port: int) -> None:
    """Serve /metrics on a daemon thread for the lifetime of the process."""
    start_http_server(port, registry=build_registry(metrics))
