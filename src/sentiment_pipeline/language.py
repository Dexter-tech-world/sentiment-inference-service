"""English-language gate.

The sentiment model is English-only: on other languages it returns confident
but meaningless predictions. This module decides whether a text is clearly
not English so it can be rejected instead of mislabelled.

The rule is deliberately asymmetric. Rejecting a real English row is worse
than letting a foreign one through (the latter just reproduces the
behaviour without a gate), so a text is only rejected on strong evidence:

1. Strip noise that confuses detectors: URLs, @mentions, #hashtags, HTML
   entities. Lowercase, squash letter runs ("yayyyyy" -> "yayy") and
   collapse whitespace.
2. Mostly non-Latin script (CJK, Cyrillic, Arabic, ...) -> not English.
3. Fewer than MIN_LETTERS letters -> undetermined -> accepted. Short texts
   ("lol ok", "Bad.") cannot be identified reliably by any detector.
4. Otherwise reject only if P(English) < MAX_ENGLISH_PROBABILITY.

Tuned on the repo's datasets: 0/3,000 UCI rows rejected, 30/30 short informal
English phrases accepted, 27/20,000 Sentiment140 tweets rejected (all of them
genuinely non-English on manual review). See README, "English language check".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import cache

MIN_LETTERS = 10
MAX_ENGLISH_PROBABILITY = 0.01
NON_LATIN_SHARE = 0.5

# Candidate languages. Restricting the set stops short English text being
# matched to obscure languages; these cover the bulk of web text.
CANDIDATE_LANGUAGES = (
    "en", "es", "fr", "de", "pt", "it", "nl", "sv", "da", "no", "pl", "tr", "id",
    "ru", "uk", "ar", "fa", "hi", "ur", "zh", "ja", "ko", "th", "vi", "el", "he",
)  # fmt: skip

_NOISE = re.compile(r"https?://\S+|www\.\S+|[@#]\w+|&\w+;")
_LETTER_RUN = re.compile(r"(.)\1{2,}")


@dataclass(frozen=True)
class LanguageVerdict:
    is_english: bool
    reason: str  # "non_latin_script" | "too_short" | "detected" | "no_letters"
    language: str | None = None  # best-guess language code, when detected
    english_probability: float | None = None

    def describe(self) -> str:
        if self.reason == "non_latin_script":
            return "text is mostly in a non-Latin script"
        return f"detected {self.language!r}, P(English)={self.english_probability:.4f}"


class EnglishDetector:
    def __init__(self) -> None:
        # py3langid: 5 MB, ~40k texts/s, deterministic. lingua was measured as
        # well: no more accurate here, 35x slower and 60x larger.
        from py3langid.langid import MODEL_FILE, LanguageIdentifier

        self._identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
        self._identifier.set_languages(list(CANDIDATE_LANGUAGES))

    def check(self, text: str) -> LanguageVerdict:
        clean = _LETTER_RUN.sub(r"\1\1", _NOISE.sub(" ", text)).lower()
        clean = " ".join(clean.split())  # padding must not change the verdict
        letters = [c for c in clean if c.isalpha()]
        if not letters:
            return LanguageVerdict(True, "no_letters")

        non_latin = sum("LATIN" not in unicodedata.name(c, "") for c in letters)
        if non_latin / len(letters) > NON_LATIN_SHARE:
            return LanguageVerdict(False, "non_latin_script")
        if len(letters) < MIN_LETTERS:
            return LanguageVerdict(True, "too_short")

        ranked = self._identifier.rank(clean)  # read-only: safe to share across threads
        english = dict(ranked).get("en", 0.0)
        best = ranked[0][0]
        return LanguageVerdict(english >= MAX_ENGLISH_PROBABILITY, "detected", best, english)


@cache
def default_detector() -> EnglishDetector:
    """One shared instance per process; loading the model takes ~0.1s."""
    return EnglishDetector()
