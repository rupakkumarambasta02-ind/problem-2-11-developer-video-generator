import copy
import pytest

from codecast.grounding.validator import ValidationContext, output_allowed, validate_submission

from tests.unit.storyboards import valid_submission


def run(repo, raw, mermaid_check=None):
    ctx = ValidationContext(repomap=repo, llm={}, output_allowlist=["*.log", "logs/**", "**/output*"],
                            mermaid_check=mermaid_check)
    return validate_submission(raw, ctx)


def codes(res):
    return {i.code for i in res.issues}


def test_scene_count_and_order(tiny_repo):
    raw = valid_submission()
    raw["scenes"] = raw["scenes"][1:4]
    assert {"SCENE_COUNT", "FIRST_SCENE", "LAST_SCENE"} <= codes(run(tiny_repo, raw))


@pytest.mark.parametrize("n_middle,ok", [(8, True), (9, False)])
def test_scene_count_upper_limit_is_ten(tiny_repo, n_middle, ok):
    raw = valid_submission()
    bullets = raw["scenes"][1]
    raw["scenes"] = [raw["scenes"][0], *[copy.deepcopy(bullets) for _ in range(n_middle)], raw["scenes"][-1]]
    for sc in raw["scenes"][1:-1]:
        sc["narration"] = sc["narration"][:1]
    assert ("SCENE_COUNT" in codes(run(tiny_repo, raw))) is not ok


def test_word_count_bounds(tiny_repo):
    raw = valid_submission()
    for sc in raw["scenes"]:
        sc["narration"] = sc["narration"][:1]
        sc["narration"][0]["text"] = "Short line here."
    assert "WORD_COUNT" in codes(run(tiny_repo, raw))


def test_sentence_rules(tiny_repo):
    raw = valid_submission()
    raw["scenes"][1]["narration"][0]["text"] = " ".join(["word"] * 31)
    raw["scenes"][1]["narration"][1]["source_ids"] = []
    raw["scenes"][2]["narration"] = []
    assert {"SENTENCE_TOO_LONG", "SENTENCE_UNCITED", "NARRATION_EMPTY"} <= codes(run(tiny_repo, raw))


def test_title_rules(tiny_repo):
    raw = valid_submission()
    raw["scenes"][0]["heading"] = "x" * 61
    raw["scenes"][0]["subheading"] = "One thing. Another thing."
    assert {"TEXT_TOO_LONG", "NOT_ONE_SENTENCE"} <= codes(run(tiny_repo, raw))


def test_bullet_limits(tiny_repo):
    raw = valid_submission()
    raw["scenes"][1]["items"] = ["a", "b", "c", "d", "e", "f"]
    assert "TOO_MANY_ITEMS" in codes(run(tiny_repo, raw))
    raw = valid_submission()
    raw["scenes"][1]["items"] = ["y" * 61]
    assert "TEXT_TOO_LONG" in codes(run(tiny_repo, raw))


def test_bullets_total_across_video(tiny_repo):
    raw = valid_submission()
    extra = {"type": "bullets", "heading": "More", "items": ["a", "b", "c", "d", "e"],
             "narration": [{"text": "More words.", "source_ids": ["readme"]}]}
    raw["scenes"][1]["items"] = ["a", "b", "c", "d", "e"]
    raw["scenes"].insert(2, dict(extra))
    raw["scenes"].insert(2, dict(extra))
    assert "TOO_MANY_BULLETS_TOTAL" in codes(run(tiny_repo, raw))


def test_code_highlight_out_of_range(tiny_repo):
    raw = valid_submission()
    raw["scenes"][2]["highlight_lines"] = [5]
    assert "HIGHLIGHT_OUT_OF_RANGE" in codes(run(tiny_repo, raw))


def test_code_too_long(tiny_repo, tmp_path):
    from pathlib import Path
    from codecast.models import FileEntry
    root = Path(tiny_repo.root)
    (root / "long.py").write_text("\n".join(f"x = {i}" for i in range(20)) + "\n")
    tiny_repo.files.append(FileEntry(path="long.py", size=1, kind="text"))
    raw = valid_submission()
    raw["sources"].append({"kind": "text", "id": "long", "file": "long.py", "start_line": 1,
                           "end_line": 19, "text": "\n".join(f"x = {i}" for i in range(19))})
    raw["scenes"][2]["span_id"] = "long"
    raw["scenes"][2]["highlight_lines"] = []
    assert "CODE_TOO_LONG" in codes(run(tiny_repo, raw))


@pytest.mark.parametrize("rect", [[0.8, 0.1, 0.3, 0.1], [-0.1, 0, 0.5, 0.5], [0.1, 0.1, 0, 0.2]])
def test_rect_bounds(tiny_repo, rect):
    raw = valid_submission()
    raw["scenes"][4]["highlights"] = [rect]
    assert "RECT_OUT_OF_BOUNDS" in codes(run(tiny_repo, raw))


def test_ken_burns_rect_bounds(tiny_repo):
    raw = valid_submission()
    raw["scenes"][4]["motion"] = {"kind": "ken_burns", "from_rect": [0, 0, 1, 1], "to_rect": [0.5, 0.5, 0.6, 0.6]}
    assert "RECT_OUT_OF_BOUNDS" in codes(run(tiny_repo, raw))


def test_table_rules(tiny_repo):
    raw = valid_submission()
    raw["scenes"][3]["rows"] = [["large@8"]]
    assert "TABLE_SHAPE" in codes(run(tiny_repo, raw))
    raw = valid_submission()
    raw["scenes"][3]["rows"][0][1] = {"span_id": "models"}
    assert "CELL_NOT_NUMERIC" in codes(run(tiny_repo, raw))
    raw = valid_submission()
    raw["scenes"][3]["rows"][0][1] = {"span_id": "f1", "decimals": 1}
    assert "CELL_UNGROUNDED" in codes(run(tiny_repo, raw))


def test_mermaid_check_used_when_available(tiny_repo):
    raw = valid_submission()
    raw["scenes"][4] = {"type": "diagram", "mermaid": "flowchart LR\n  A --> B",
                        "narration": [{"text": "Flow.", "source_ids": ["readme"]}]}
    assert "MERMAID_SYNTAX" in codes(run(tiny_repo, raw, mermaid_check=lambda code: "Parse error"))
    assert "MERMAID_SYNTAX" not in codes(run(tiny_repo, raw, mermaid_check=lambda code: None))


def test_terminal_rules(tiny_repo):
    raw = valid_submission()
    raw["sources"].append({"kind": "text", "id": "out", "file": "README.md", "start_line": 1,
                           "end_line": 1, "text": "# Tiny"})
    raw["sources"].append({"kind": "text", "id": "cmd", "file": "README.md", "start_line": 3,
                           "end_line": 3, "text": "Run `python app.py --top 3` to predict."})
    raw["scenes"][4] = {"type": "terminal", "command_span_ids": ["cmd"], "output_span_id": "out",
                        "narration": [{"text": "Run it.", "source_ids": ["cmd"]}]}
    assert "OUTPUT_NOT_ALLOWED" in codes(run(tiny_repo, raw))
    raw["scenes"][4]["command_span_ids"] = []
    assert "TERMINAL_NO_COMMANDS" in codes(run(tiny_repo, raw))


def test_outro_repo_url_must_match(tiny_repo):
    raw = valid_submission()
    raw["scenes"][-1]["repo_url"] = "https://github.com/evil/fork"
    assert "REPO_URL_MISMATCH" in codes(run(tiny_repo, raw))


@pytest.mark.parametrize("path,ok", [
    ("logs/run.log", True), ("run.log", True), ("a/b/output.txt", True), ("logs/x/y.json", True),
    ("README.md", False), ("src/app.py", False),
])
def test_output_allowlist(path, ok):
    assert output_allowed(path, ["*.log", "logs/**", "**/output*"]) is ok


def test_valid_submission_still_accepted(tiny_repo):
    assert run(tiny_repo, valid_submission()).issues == []


def _code_span(tiny_repo, lines):
    from pathlib import Path
    from codecast.models import FileEntry
    (Path(tiny_repo.root) / "wide.jsx").write_text("\n".join(lines) + "\n")
    tiny_repo.files.append(FileEntry(path="wide.jsx", size=1, kind="text"))
    raw = valid_submission()
    raw["sources"].append({"kind": "text", "id": "wide", "file": "wide.jsx", "start_line": 1,
                           "end_line": len(lines)})
    raw["scenes"][2]["span_id"] = "wide"
    raw["scenes"][2]["highlight_lines"] = []
    return run(tiny_repo, raw)


@pytest.mark.parametrize("lines,too_wide", [
    (["x" * 141, "y = 1"], True),
    (["x" * 140, "y = 1"], False),
    ([" " * 10 + "x" * 139, " " * 12 + "y = 1"], False),   # shared indent is dropped before measuring
    (["\t\t" + "x" * 130, "y = 1"], True),                  # tabs count as 8 columns
])
def test_code_width_limit(tiny_repo, lines, too_wide):
    res = _code_span(tiny_repo, lines)
    issues = [i for i in res.issues if i.code == "CODE_TOO_WIDE"]
    assert bool(issues) is too_wide
    if too_wide:
        assert issues[0].path == "scenes[2].span_id" and "max 140" in issues[0].message
