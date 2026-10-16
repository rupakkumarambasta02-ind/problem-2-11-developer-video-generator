import pytest

from codecast.grounding.numbers import (
    Claim, classify, extract_claims, normalize_text, source_number_literals, source_tokens,
    split_tokens, words_to_digits,
)


def test_unicode_minus_normalized():
    assert normalize_text("−0.5") == "-0.5"


@pytest.mark.parametrize("text,expected", [
    ("six classes", "6 classes"),
    ("Twenty-one runs", "21 runs"),
    ("one hundred and five clips", "105 clips"),
    ("two thousand three hundred", "2300"),
    ("three million", "3000000"),
    ("the first model", "the first model"),        # ordinals untouched
    ("a one-line command", "a 1-line command"),
    ("seventeen", "17"),
    ("ninety nine", "99"),
])
def test_words_to_digits(text, expected):
    assert words_to_digits(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("between one and two", "between 1 and 2"),
    ("one two three", "1 2 3"),
    ("four and two", "4 and 2"),
    ("one thousand and five", "1005"),
    ("twenty thirty", "20 30"),
    ("two three-line tests", "2 3-line tests"),
    ("twenty-one and six", "21 and 6"),
])
def test_words_to_digits_keeps_separate_numbers(text, expected):
    assert words_to_digits(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("F1 is 0.735.", ["F1", "is", "0.735"]),
    ("We used ViT-B-32 (large@8).", ["We", "used", "ViT-B-32", "large@8"]),
    ("73.5% of 1,234 clips", ["73.5%", "of", "1,234", "clips"]),
    ("3,4 and 1,23456", ["3", "4", "and", "1", "23456"]),
    ("see src/metrics.py:12", ["see", "src/metrics.py", "12"]),
    ("emoji 🎬 and <b>", ["emoji", "and", "b"]),
    ("a,123 x", ["a", "123", "x"]),
])
def test_split_tokens(text, expected):
    assert split_tokens(text) == expected


@pytest.mark.parametrize("token,kind", [
    ("0.735", "number"), ("-2", "number"), ("1,234", "number"), ("73.5%", "number"),
    ("ViT-B-32", "identifier"), ("large@8", "identifier"), ("1k", "identifier"),
    ("2e3", "identifier"), ("MobileCLIP2-S0", "identifier"), ("1.2.3", "identifier"),
    ("model", None), ("src/app.py", None),
])
def test_classify(token, kind):
    assert classify(token) == kind


def test_extract_claims_converts_number_words_only_for_checking():
    claims = extract_claims("Six classes, scored by ViT-B-32 at 0.73.")
    assert claims == [Claim("number", "6"), Claim("identifier", "ViT-B-32"), Claim("number", "0.73")]


def test_source_side_helpers():
    text = "f1: 0.7349, latency 891.2 ms with large@8 and six folds"
    assert source_number_literals(text) == ["0.7349", "891.2", "6"]
    assert "large@8" in source_tokens(text)
