"""Sentiment model interface and the Hugging Face implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

DEFAULT_MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
# Pinned commit so a model update on the Hub can't silently change predictions.
DEFAULT_MODEL_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


class SentimentModel(Protocol):
    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Return one Prediction per input text, in the same order."""
        ...


class HuggingFaceSentimentModel:
    """A sequence-classification model run with plain PyTorch on CPU."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str | None = DEFAULT_MODEL_REVISION,
        max_length: int = 512,
    ):
        # Imported here so the rest of the package (and its tests) don't need torch.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision
        ).eval()
        self.id2label = {int(i): label.lower() for i, label in self.model.config.id2label.items()}

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        # Truncate rather than fail: inputs longer than the model's context are
        # classified on their first `max_length` tokens.
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            logits = self.model(**encoded).logits
        scores, indices = logits.softmax(dim=-1).max(dim=-1)
        return [
            Prediction(label=self.id2label[i], score=round(s, 4))
            for i, s in zip(indices.tolist(), scores.tolist(), strict=True)
        ]
