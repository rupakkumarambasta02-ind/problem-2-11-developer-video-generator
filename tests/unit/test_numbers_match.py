import pytest

from codecast.grounding.numbers import decimals, identifier_grounded, number_grounded


@pytest.mark.parametrize("value,sources,ok", [
    # §7.2 examples, source 0.7349
    ("0.7349", ["0.7349"], True),
    ("0.735", ["0.7349"], True),
    ("0.73", ["0.7349"], True),
    ("73.5%", ["0.7349"], True),
    ("73%", ["0.7349"], True),
    ("73.49%", ["0.7349"], True),
    ("0.7", ["0.7349"], False),      # 1 significant figure
    ("0.7350", ["0.7349"], False),   # more decimals than source
    ("891", ["891.2"], True),
    # equality and decimals
    ("42", ["42"], True),
    ("0.70", ["0.7"], False),
    ("1,234", ["1234"], True),
    ("1234", ["1,234"], True),
    # half-even vs half-up both accepted
    ("0.12", ["0.125"], True),       # half-even
    ("0.13", ["0.125"], True),       # half-up
    # percent rules
    ("74%", ["73.5%"], True),
    ("73.5", ["73.5%"], False),      # non-percent narration vs percent source
    ("7%", ["0.07"], False),         # 1 significant figure
    ("68%", ["0.677"], True),
    ("67.7%", ["0.677"], True),
    ("67.70%", ["0.677"], False),
    # derived values never match
    ("1311", ["420", "891"], False),
    ("-2", ["-2"], True),
    ("5", [], False),
])
def test_number_grounded(value, sources, ok):
    assert number_grounded(value, sources) is ok


def test_decimals():
    assert decimals("0.7349") == 4
    assert decimals("891") == 0
    assert decimals("73.5%") == 1
    assert decimals("1,234.50") == 2


def test_identifier_grounding_is_case_sensitive_whole_token():
    texts = ["encoder: ViT-B-32 (laion2b_s34b_b79k)", "head large@8"]
    assert identifier_grounded("ViT-B-32", texts)
    assert identifier_grounded("large@8", texts)
    assert not identifier_grounded("vit-b-32", texts)
    assert not identifier_grounded("ViT-B", texts)     # not a whole token
    assert not identifier_grounded("B-32", texts)


def test_long_source_literal_does_not_raise():
    # Would overflow the default 28-digit context in quantize without a sized context.
    assert number_grounded("12", ["1234567890123456789012345678901.5"]) is False


def test_long_literal_legitimately_rounds():
    assert number_grounded(
        "1234567890123456789012345678901234",
        ["1234567890123456789012345678901234.4"],
    ) is True
