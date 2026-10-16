import json
from pathlib import Path

import pytest

from codecast.agent.tools import SubmitOutcome, ToolBox
from codecast.errors import ValidationFailed
from codecast.grounding.spans import check_sources
from codecast.llm.types import ImagePart, TextPart, ToolCallPart
from codecast.models import FileEntry, SkippedEntry, TextSpan


def box(repo, submit=None):
    return ToolBox(repo, submit or (lambda raw: SubmitOutcome(False, None, "rejected")))


def run(tb, name, **args):
    ex = tb.execute(ToolCallPart("c1", name, args))
    text = "\n".join(p.text for p in ex.result.parts if isinstance(p, TextPart))
    return ex, text


def test_specs_drop_view_image_without_vision(tiny_repo):
    names = [s.name for s in box(tiny_repo).specs(vision=False)]
    assert names == ["list_dir", "read_file", "grep", "read_json", "submit_storyboard"]
    assert "view_image" in [s.name for s in box(tiny_repo).specs(vision=True)]


def test_submit_schema_has_hoisted_defs(tiny_repo):
    spec = {s.name: s for s in box(tiny_repo).specs(vision=True)}["submit_storyboard"]
    assert "$defs" in spec.input_schema
    assert "$defs" not in spec.input_schema["properties"]["storyboard"]


def test_list_dir(tiny_repo):
    ex, text = run(box(tiny_repo), "list_dir", path=".")
    data = json.loads(text)
    paths = [e["path"] for e in data["entries"]]
    assert "src/" in paths and "README.md" in paths and "src/app.py" not in paths
    _, text2 = run(box(tiny_repo), "list_dir", path=".", depth=2)
    assert "src/app.py" in text2


def test_read_file_numbered_and_paged(tiny_repo):
    ex, text = run(box(tiny_repo), "read_file", path="README.md", start_line=2, max_lines=1)
    data = json.loads(text)
    assert data["lines"] == ["2| Tiny classifies clips with ViT-B-32 at 0.7349 F1."]
    assert data["truncated"] is True and data["next_start_line"] == 3 and data["total_lines"] == 3


def test_read_file_output_round_trips_into_span(tiny_repo):
    """Review Focus 2: CRLF + BOM file, text copied from read_file passes span integrity."""
    _, text = run(box(tiny_repo), "read_file", path="crlf.txt")
    lines = [l.split("| ", 1)[1] for l in json.loads(text)["lines"]]
    span = TextSpan(id="s", file="crlf.txt", start_line=1, end_line=2, text="\n".join(lines))
    issues, _ = check_sources([span], tiny_repo)
    assert issues == []


def test_read_file_caps_bytes(tiny_repo):
    root = Path(tiny_repo.root)
    (root / "wide.txt").write_text("\n".join("x" * 1000 for _ in range(100)))
    tiny_repo.files.append(FileEntry(path="wide.txt", size=1, kind="text"))
    _, text = run(box(tiny_repo), "read_file", path="wide.txt")
    data = json.loads(text)
    assert len(text.encode()) <= 32 * 1024
    assert data["truncated"] and data["next_start_line"] == len(data["lines"]) + 1


def test_read_file_caps_escape_heavy_lines(tiny_repo):
    root = Path(tiny_repo.root)
    (root / "esc.txt").write_text("\n".join('"\\\\\\"' * 100 for _ in range(200)) + "\n")
    tiny_repo.files.append(FileEntry(path="esc.txt", size=1, kind="text"))
    _, text = run(box(tiny_repo), "read_file", path="esc.txt")
    data = json.loads(text)
    assert len(text.encode()) <= 32 * 1024
    assert data["truncated"] is True
    assert data["next_start_line"] == int(data["lines"][-1].split("|", 1)[0]) + 1


def test_read_file_single_huge_quote_line_fits(tiny_repo):
    root = Path(tiny_repo.root)
    (root / "huge.txt").write_text('"' * 60000 + "\n")
    tiny_repo.files.append(FileEntry(path="huge.txt", size=1, kind="text"))
    ex, text = run(box(tiny_repo), "read_file", path="huge.txt")
    data = json.loads(text)
    assert len(text.encode()) <= 32 * 1024
    assert data["lines"][0].endswith("[line truncated]") and data["truncated"] is True


def test_grep_output_capped(tiny_repo):
    root = Path(tiny_repo.root)
    (root / "many.txt").write_text("\n".join('"\\' * 100 for _ in range(400)) + "\n")
    tiny_repo.files.append(FileEntry(path="many.txt", size=1, kind="text"))
    _, text = run(box(tiny_repo), "grep", pattern=r'"', max_results=200)
    data = json.loads(text)
    assert len(text.encode()) <= 32 * 1024
    assert data["truncated"] is True and len(data["matches"]) < 200


@pytest.mark.parametrize("args,needle", [
    ({"path": "../outside"}, "escapes"),
    ({"path": "plot.png"}, "image"),
    ({"path": "missing.md"}, "not a file"),
])
def test_read_file_errors(tiny_repo, args, needle):
    ex, text = run(box(tiny_repo), "read_file", **args)
    assert ex.result.is_error and needle in text


def test_read_file_reports_skip_reason(tiny_repo):
    tiny_repo.skipped.append(SkippedEntry(path="big.csv", reason="too_large"))
    ex, text = run(box(tiny_repo), "read_file", path="big.csv")
    assert ex.result.is_error and "too_large" in text


def test_grep(tiny_repo):
    _, text = run(box(tiny_repo), "grep", pattern=r"ViT-\w", path_glob="*.md")
    data = json.loads(text)
    assert data["matches"] == [{"file": "README.md", "line": 2,
                                "text": "Tiny classifies clips with ViT-B-32 at 0.7349 F1."}]
    ex, text = run(box(tiny_repo), "grep", pattern="(")
    assert ex.result.is_error and "regex" in text


def test_read_json_pointer(tiny_repo):
    _, text = run(box(tiny_repo), "read_json", path="metrics.json", pointer="/models/large@8")
    assert '"f1": 0.7349' in text and "891.2" in text
    ex, text = run(box(tiny_repo), "read_json", path="metrics.json", pointer="/nope")
    assert ex.result.is_error and "not found" in text


def test_view_image(tiny_repo):
    ex = box(tiny_repo).execute(ToolCallPart("c1", "view_image", {"path": "plot.png"}))
    img = [p for p in ex.result.parts if isinstance(p, ImagePart)][0]
    assert img.media_type == "image/png" and img.source_path == "plot.png"
    assert "400x300" in ex.result.parts[0].text


@pytest.mark.parametrize("name,args,needle", [
    ("nope", {}, "unknown tool"),
    ("read_file", {}, "missing required argument 'path'"),
    ("read_file", {"path": "README.md", "colour": 1}, "unexpected argument 'colour'"),
    ("read_file", {"path": "README.md", "start_line": "abc"}, "'start_line' must be integer"),
    ("grep", {"__invalid_json__": "{bad"}, "not valid JSON"),
])
def test_malformed_calls_become_errors(tiny_repo, name, args, needle):
    """Review Focus 5."""
    ex = box(tiny_repo).execute(ToolCallPart("c1", name, args))
    assert ex.result.is_error and needle in ex.result.parts[0].text


def test_submit_passes_raw_and_parses_json_string(tiny_repo):
    seen = []

    def submit(raw):
        seen.append(raw)
        return SubmitOutcome(True, "SB", "accepted")
    ex, text = run(box(tiny_repo, submit), "submit_storyboard", storyboard='{"title": "x"}')
    assert seen == [{"title": "x"}] and ex.storyboard == "SB" and not ex.result.is_error and text == "accepted"
    ex, _ = run(box(tiny_repo, submit), "submit_storyboard", storyboard="not json")
    assert seen[-1] == "not json"


def test_validation_failed_propagates(tiny_repo):
    def submit(raw):
        raise ValidationFailed("4 rounds")
    with pytest.raises(ValidationFailed):
        box(tiny_repo, submit).execute(ToolCallPart("c1", "submit_storyboard", {"storyboard": {}}))


def test_deeply_nested_json_becomes_error(tiny_repo):
    """Controller note: RecursionError from load_json_decimal must not crash the run."""
    (Path(tiny_repo.root) / "deep.json").write_text("[" * 100000 + "]" * 100000)
    tiny_repo.files.append(FileEntry(path="deep.json", size=1, kind="text"))
    ex, text = run(box(tiny_repo), "read_json", path="deep.json")
    assert ex.result.is_error


@pytest.mark.parametrize("args", [{"path": "README.md", "start_line": 0}, {"path": "README.md", "max_lines": 0}])
def test_out_of_range_numbers_become_errors(tiny_repo, args):
    ex, text = run(box(tiny_repo), "read_file", **args)
    assert ex.result.is_error


def test_read_json_reads_yaml(tiny_repo):
    root = Path(tiny_repo.root)
    (root / "board.yml").write_text("- model: gemini\n  pass_rate_1: 20.4\n")
    tiny_repo.files.append(FileEntry(path="board.yml", size=40, kind="text"))
    ex, text = run(box(tiny_repo), "read_json", path="board.yml", pointer="/0/pass_rate_1")
    assert not ex.result.is_error and json.loads(text)["value"] == 20.4
    (root / "bad.yml").write_text("a: [1, 2\n")
    tiny_repo.files.append(FileEntry(path="bad.yml", size=9, kind="text"))
    ex, text = run(box(tiny_repo), "read_json", path="bad.yml")
    assert ex.result.is_error and "YAML" in text


def test_read_json_accepts_comments_and_hints_slash_keys(tiny_repo):
    root = Path(tiny_repo.root)
    (root / "meta.json").write_text('{\n  // c\n  "a/b": {"n": 1},\n}\n')
    tiny_repo.files.append(FileEntry(path="meta.json", size=30, kind="text"))
    ex, text = run(box(tiny_repo), "read_json", path="meta.json", pointer="/a~1b/n")
    assert not ex.result.is_error and json.loads(text)["value"] == 1
    ex, text = run(box(tiny_repo), "read_json", path="meta.json", pointer="/a/b/n")
    assert ex.result.is_error and "/a~1b" in text
