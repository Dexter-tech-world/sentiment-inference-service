"""Validation and fault-isolating inference, shared by the batch pipeline and the HTTP API.

Keeping these in one place guarantees both entry points accept, reject and
classify text identically.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .language import EnglishDetector
from .metrics import EMPTY_TEXT, NON_ENGLISH, Metrics
from .model import Prediction, SentimentModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rejection:
    reason: str  # one of the bounded failure reasons in metrics.py
    detail: str


def validate_text(text: str | None, detector: EnglishDetector | None) -> Rejection | None:
    """Return why `text` must not be sent to the model, or None if it may be."""
    if text is None or not text.strip():
        return Rejection(EMPTY_TEXT, "text is empty")
    if detector is not None:
        verdict = detector.check(text)
        if not verdict.is_english:
            return Rejection(NON_ENGLISH, verdict.describe())
    return None


def predict_with_fallback(
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
