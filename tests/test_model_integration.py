"""Runs the real model. Opt-in because it needs torch and a ~270MB download.

RUN_MODEL_TESTS=1 pytest tests/test_model_integration.py
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to run"
)


@pytest.fixture(scope="module")
def model():
    from sentiment_pipeline.model import HuggingFaceSentimentModel

    return HuggingFaceSentimentModel()


def test_obvious_sentences_are_classified_correctly(model):
    predictions = model.predict(["I love this, it is fantastic.", "This is awful and broken."])
    assert [p.label for p in predictions] == ["positive", "negative"]
    assert all(0.5 <= p.score <= 1.0 for p in predictions)


def test_text_longer_than_model_context_is_truncated_not_failed(model):
    (prediction,) = model.predict(["great movie " * 2000])
    assert prediction.label == "positive"
