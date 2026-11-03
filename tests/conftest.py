from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import pytest

from sentiment_pipeline.model import Prediction


class FakeModel:
    """Deterministic stand-in: 'positive' if the text contains 'good', else 'negative'.

    Any text containing a `poison` substring makes predict() raise, which lets
    tests exercise batch failure and row-level isolation.
    """

    def __init__(self, poison: Sequence[str] = (), always_fail: bool = False):
        self.poison = tuple(poison)
        self.always_fail = always_fail
        self.calls: list[list[str]] = []

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        self.calls.append(list(texts))
        if self.always_fail:
            raise RuntimeError("model is broken")
        for text in texts:
            if any(p in text for p in self.poison):
                raise ValueError(f"cannot handle {text!r}")
        return [Prediction("positive" if "good" in t else "negative", 0.9) for t in texts]


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def read_output(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
