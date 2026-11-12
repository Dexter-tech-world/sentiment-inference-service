from __future__ import annotations

import pytest

from sentiment_pipeline.language import default_detector


@pytest.fixture(scope="module")
def detector():
    return default_detector()


# Rejecting real English is the expensive mistake, so most cases guard against it.
@pytest.mark.parametrize(
    "text",
    [
        "lol ok",  # too short to judge: accepted
        "Bad.",
        "@john thx!",
        "LOL yayyyyy!",  # letter runs are squashed before detection
        "The WORST EXPERIENCE EVER.",  # case is normalised
        "Penne vodka excellent!",
        "wow i so dumb.",
        "The Da Vinci Code!! eek.",
        "I absolutely loved this phone. Battery lasts forever.",
        "@someone check this out http://example.com/very/long/path it is great",
        "1234 !!! ???",  # no letters at all
    ],
)
def test_english_is_accepted(detector, text):
    verdict = detector.check(text)
    assert verdict.is_english, verdict


@pytest.mark.parametrize(
    "text",
    [
        "Ce film était magnifique",
        "Das Essen war schrecklich",
        "Il servizio è stato pessimo",
        "@cliente Não gostei nada deste produto, muito ruim",  # mention is stripped first
        "El servicio fue terrible y muy lento",
    ],
)
def test_clearly_foreign_text_is_rejected(detector, text):
    verdict = detector.check(text)
    assert not verdict.is_english
    assert verdict.reason == "detected"
    assert verdict.english_probability < 0.01


@pytest.mark.parametrize(
    "text", ["この映画は最高でした", "这个产品很好", "هذا المنتج رائع", "Этот фильм ужасен"]
)
def test_non_latin_script_is_rejected_regardless_of_length(detector, text):
    verdict = detector.check(text)
    assert not verdict.is_english
    assert verdict.reason == "non_latin_script"


def test_verdict_explains_itself(detector):
    assert "P(English)" in detector.check("Das Essen war schrecklich").describe()
    assert "non-Latin" in detector.check("这个产品很好").describe()


def test_whitespace_padding_does_not_change_the_verdict(detector):
    text = "Sin internet en el curre"  # short text near the threshold
    assert (
        detector.check(text)
        == detector.check(f"  {text} \n")
        == detector.check(text.replace(" ", "   "))
    )
