from decimal import Decimal
from pathlib import Path

import pytest

from codecast.grounding.spans import (
    PointerError, check_sources, dumps_decimal, format_cell, load_data_decimal, load_json_decimal, read_lines,
    resolve_pointer, values_equal,
)
from codecast.models import FileEntry, ImageRef, JsonSpan, TextSpan


@pytest.mark.parametrize("pointer,expected", [
    ("", {"a/b": 1, "m~n": [10, 20], "x": {"y": None}}),
    ("/a~1b", 1),
    ("/m~0n/1", 20),
    ("/x/y", None),
])
def test_resolve_pointer(pointer, expected):
    doc = {"a/b": 1, "m~n": [10, 20], "x": {"y": None}}
    assert resolve_pointer(doc, pointer) == expected


@pytest.mark.parametrize("pointer", ["a", "/missing", "/m~0n/2", "/m~0n/01", "/a~1b/c"])
def test_resolve_pointer_errors(pointer):
    with pytest.raises(PointerError):
        resolve_pointer({"a/b": 1, "m~n": [10, 20]}, pointer)


def test_decimal_json_round_trip(tiny_repo):
    doc = load_json_decimal(Path(tiny_repo.root) / "metrics.json")
    assert doc["models"]["large@8"]["f1"] == Decimal("0.7349")
    text = dumps_decimal(doc["models"])
    assert "0.7349" in text and "891.2" in text and "large@8" in text


def test_crlf_bom_and_invalid_utf8_lines(tiny_repo):
    root = Path(tiny_repo.root)
    assert read_lines(root / "crlf.txt") == ["alpha 1", "beta 2"]
    assert read_lines(root / "bad.txt")[0] == "ok"


def test_values_equal():
    assert values_equal(0.7349, Decimal("0.7349"))
    assert values_equal(420, 420)
    assert not values_equal(0.735, Decimal("0.7349"))
    assert not values_equal(True, 1)
    assert values_equal({"a": [1.5]}, {"a": [Decimal("1.5")]})


def test_check_sources_happy_path(tiny_repo):
    sources = [
        TextSpan(id="r", file="README.md", start_line=2, end_line=2,
                 text="Tiny classifies clips with ViT-B-32 at 0.7349 F1."),
        TextSpan(id="c", file="crlf.txt", start_line=1, end_line=2, text="alpha 1\nbeta 2\n"),
        JsonSpan(id="j", file="metrics.json", pointer="/models/large@8/f1", value=0.7349),
        ImageRef(id="i", file="plot.png"),
    ]
    issues, index = check_sources(sources, tiny_repo)
    assert issues == []
    assert index.values["j"] == Decimal("0.7349")
    assert index.texts["j"] == "0.7349"
    assert "ViT-B-32" in index.texts["r"]
    assert index.texts["i"] == ""


def test_check_sources_reports_every_failure(tiny_repo):
    sources = [
        TextSpan(id="a", file="README.md", start_line=2, end_line=2, text="invented text"),
        TextSpan(id="a", file="README.md", start_line=1, end_line=1, text="# Tiny"),
        TextSpan(id="b", file="nope.md", start_line=1, end_line=1, text="x"),
        TextSpan(id="c", file="README.md", start_line=3, end_line=9, text="x"),
        TextSpan(id="d", file="plot.png", start_line=1, end_line=1, text="x"),
        JsonSpan(id="e", file="metrics.json", pointer="/models/huge", value=1),
        JsonSpan(id="f", file="metrics.json", pointer="/n_videos", value=91),
        JsonSpan(id="g", file="README.md", pointer="", value=1),
        ImageRef(id="h", file="README.md"),
    ]
    issues, _ = check_sources(sources, tiny_repo)
    codes = sorted(i.code for i in issues)
    assert codes == sorted([
        "SPAN_TEXT_MISMATCH", "DUPLICATE_SOURCE_ID", "SPAN_FILE_MISSING", "SPAN_RANGE_INVALID",
        "SPAN_NOT_TEXT", "JSON_POINTER_UNRESOLVED", "JSON_VALUE_MISMATCH", "NOT_A_DATA_FILE",
        "IMAGE_NOT_FOUND",
    ])
    assert all(i.path.startswith("sources[") for i in issues)


@pytest.mark.parametrize("value,dec,expected", [
    (Decimal("0.7349"), None, "0.7349"),
    (Decimal("0.7349"), 3, "0.735"),
    (420, None, "420"),
    (Decimal("891.2"), 0, "891"),
    ("text", None, None),
    (True, None, None),
])
def test_format_cell(value, dec, expected):
    assert format_cell(value, dec) == expected


def test_deep_nested_json_is_a_parse_error_not_a_crash(tiny_repo):
    from codecast.models import FileEntry
    root = Path(tiny_repo.root)
    (root / "deep.json").write_text("[" * 200000)
    tiny_repo.files.append(FileEntry(path="deep.json", size=200000, kind="text"))
    issues, _ = check_sources([JsonSpan(id="d", file="deep.json", pointer="", value=1)], tiny_repo)
    assert [i.code for i in issues] == ["JSON_PARSE_ERROR"]


def test_span_file_deleted_after_repomap_is_file_missing(tiny_repo):
    (Path(tiny_repo.root) / "README.md").unlink()
    issues, _ = check_sources(
        [TextSpan(id="r", file="README.md", start_line=1, end_line=1, text="# Tiny")], tiny_repo)
    assert [i.code for i in issues] == ["SPAN_FILE_MISSING"]


def test_resolve_pointer_huge_index_is_pointer_error():
    with pytest.raises(PointerError):
        resolve_pointer([1], "/" + "9" * 5000)


def test_values_equal_huge_int_does_not_raise():
    assert values_equal(10**5000, 10**5000)
    assert not values_equal(10**5000, 1)


def test_format_cell_non_finite_and_negative_decimals():
    assert format_cell(Decimal("Infinity"), 2) is None
    assert format_cell(Decimal("NaN"), 2) is None
    assert format_cell(Decimal("1.5"), -1) is None


def test_format_cell_wide_value_regression():
    assert format_cell(Decimal("1e30"), 3) == "1000000000000000000000000000000.000"


def test_span_mismatch_message_pinpoints_line_count(tiny_repo):
    issues, _ = check_sources([TextSpan(id="c", file="crlf.txt", start_line=1, end_line=2, text="alpha one")],
                              tiny_repo)
    assert issues[0].code == "SPAN_TEXT_MISMATCH"
    assert "1 line(s)" in issues[0].message and "lines 1-2 are 2 lines" in issues[0].message


def test_span_mismatch_message_shows_first_differing_file_line(tiny_repo):
    issues, _ = check_sources([TextSpan(id="c", file="crlf.txt", start_line=1, end_line=2,
                                        text="alpha 1\nbeta  2")], tiny_repo)
    assert issues[0].code == "SPAN_TEXT_MISMATCH"
    assert "line 2" in issues[0].message and "'beta 2'" in issues[0].message


def _with_blank_lines(repo):
    (Path(repo.root) / "blank.md").write_text("a\n\nb\n")
    repo.files.append(FileEntry(path="blank.md", size=5, kind="text"))
    return repo


@pytest.mark.parametrize("text", ["a\n", "a\n\n"])
def test_span_ending_on_blank_line_matches_with_or_without_terminal_newline(tiny_repo, text):
    repo = _with_blank_lines(tiny_repo)
    issues, index = check_sources([TextSpan(id="s", file="blank.md", start_line=1, end_line=2, text=text)], repo)
    assert issues == []
    assert index.texts["s"] == "a\n"


@pytest.mark.parametrize("text", ["a", "a\n\n\n"])
def test_span_ending_on_blank_line_still_rejects_wrong_line_count(tiny_repo, text):
    repo = _with_blank_lines(tiny_repo)
    issues, _ = check_sources([TextSpan(id="s", file="blank.md", start_line=1, end_line=2, text=text)], repo)
    assert [i.code for i in issues] == ["SPAN_TEXT_MISMATCH"]


@pytest.mark.parametrize("start,end,first,last", [(1, 2, 1, 3), (1, 2, 2, 3), (2, 3, 1, 2)])
def test_span_mismatch_message_names_the_lines_the_text_really_is(tiny_repo, start, end, first, last):
    lines = read_lines(Path(tiny_repo.root) / "README.md")
    text = "\n".join(lines[first - 1:last])
    issues, _ = check_sources([TextSpan(id="r", file="README.md", start_line=start, end_line=end, text=text)],
                              tiny_repo)
    assert issues[0].code == "SPAN_TEXT_MISMATCH"
    assert f"your text is exactly lines {first}-{last}" in issues[0].message


def test_text_span_without_text_is_filled_exactly(tiny_repo):
    spans = [TextSpan(id="a", file="src/app.py", start_line=1, end_line=2),
             TextSpan(id="c", file="crlf.txt", start_line=1, end_line=2)]
    issues, index = check_sources(spans, tiny_repo)
    assert issues == []
    assert index.spans["a"].text == 'def main():\n    print("hello")'
    assert index.spans["c"].text == "alpha 1\nbeta 2"
    assert index.texts["a"] == index.spans["a"].text


def test_text_span_provided_but_wrong_still_mismatches(tiny_repo):
    issues, _ = check_sources([TextSpan(id="a", file="src/app.py", start_line=1, end_line=2, text="nope")], tiny_repo)
    assert [i.code for i in issues] == ["SPAN_TEXT_MISMATCH"]


def test_json_span_without_value_is_filled(tiny_repo):
    spans = [JsonSpan(id="f", file="metrics.json", pointer="/models/large@8/f1"),
             JsonSpan(id="m", file="metrics.json", pointer="/models")]
    issues, index = check_sources(spans, tiny_repo)
    assert issues == []
    assert index.spans["f"].value == 0.7349 and "value" in index.spans["f"].model_fields_set
    assert values_equal(index.spans["m"].value, index.values["m"])
    assert index.spans["m"].value["large@8"]["latency_ms"] == 891.2


def test_json_span_explicit_null_is_still_compared(tiny_repo):
    issues, _ = check_sources([JsonSpan(id="f", file="metrics.json", pointer="/models/large@8/f1", value=None)], tiny_repo)
    assert [i.code for i in issues] == ["JSON_VALUE_MISMATCH"]


LEADERBOARD_YML = """\
- dirname: 2025-02-25-20-23-07--gemini-pro
  model: Gemini 2.0 Pro exp-02-05
  pass_rate_1: 20.4
  pass_num_1: 46
  total_cost: 0.0000
  date: 2025-02-25
"""


def _add_yaml(repo, name="board.yml", text=LEADERBOARD_YML):
    root = Path(repo.root)
    (root / name).write_text(text)
    repo.files.append(FileEntry(path=name, size=len(text), kind="text"))
    return root / name


def test_yaml_loads_floats_as_exact_decimals_and_dates_as_strings(tiny_repo):
    doc = load_data_decimal(_add_yaml(tiny_repo))
    assert doc[0]["pass_rate_1"] == Decimal("20.4")
    assert str(doc[0]["total_cost"]) == "0.0000"
    assert doc[0]["pass_num_1"] == 46
    assert doc[0]["date"] == "2025-02-25"


@pytest.mark.parametrize("name", ["board.yml", "board.yaml"])
def test_json_span_resolves_into_yaml(tiny_repo, name):
    _add_yaml(tiny_repo, name)
    spans = [JsonSpan(id="p", file=name, pointer="/0/pass_rate_1"),
             JsonSpan(id="row", file=name, pointer="/0")]
    issues, index = check_sources(spans, tiny_repo)
    assert issues == []
    assert index.values["p"] == Decimal("20.4") and index.texts["p"] == "20.4"
    assert index.spans["row"].value["date"] == "2025-02-25"


def test_malformed_yaml_is_a_parse_error(tiny_repo):
    _add_yaml(tiny_repo, text="a: [1, 2\n")
    issues, _ = check_sources([JsonSpan(id="p", file="board.yml", pointer="/a")], tiny_repo)
    assert [i.code for i in issues] == ["JSON_PARSE_ERROR"]
    assert "YAML" in issues[0].message


JSONC = """\
{
    // line comment
    "deepseek/deepseek-reasoner": {
        "max_input_tokens": 128000,
        //"supports_function_calling": true,
        "url": "https://example.com/a//b", /* block
        comment */ "note": "has /* not a comment */ inside",
        "cost": 0.00000028,
    },
}
"""


def test_json_with_comments_and_trailing_commas_loads(tiny_repo):
    path = _add_yaml(tiny_repo, "meta.json", JSONC)
    doc = load_data_decimal(path)
    entry = doc["deepseek/deepseek-reasoner"]
    assert entry["max_input_tokens"] == 128000 and entry["cost"] == Decimal("0.00000028")
    assert entry["url"] == "https://example.com/a//b"
    assert entry["note"] == "has /* not a comment */ inside"
    assert "supports_function_calling" not in entry


def test_broken_json_still_fails_even_after_comment_stripping(tiny_repo):
    _add_yaml(tiny_repo, "bad.json", '{"a": 1 // missing brace\n')
    issues, _ = check_sources([JsonSpan(id="b", file="bad.json", pointer="/a")], tiny_repo)
    assert [i.code for i in issues] == ["JSON_PARSE_ERROR"]


def test_pointer_hint_for_unescaped_slash_key(tiny_repo):
    _add_yaml(tiny_repo, "meta.json", JSONC)
    issues, _ = check_sources([JsonSpan(id="m", file="meta.json",
                                        pointer="/deepseek/deepseek-reasoner/max_input_tokens")], tiny_repo)
    assert [i.code for i in issues] == ["JSON_POINTER_UNRESOLVED"]
    assert "/deepseek~1deepseek-reasoner" in issues[0].message
    issues, index = check_sources([JsonSpan(id="m", file="meta.json",
                                            pointer="/deepseek~1deepseek-reasoner/max_input_tokens")], tiny_repo)
    assert issues == [] and index.values["m"] == 128000


def test_json_source_on_non_data_file_is_rejected_with_hint(tiny_repo):
    issues, _ = check_sources([JsonSpan(id="r", file="README.md", pointer="/0/x")], tiny_repo)
    assert [i.code for i in issues] == ["NOT_A_DATA_FILE"]
    assert "kind text" in issues[0].message and "README.md" in issues[0].message
