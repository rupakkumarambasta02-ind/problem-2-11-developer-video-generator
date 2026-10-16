import pytest
from pydantic import ValidationError

from codecast.models import (
    CellRef, FileEntry, ImageScene, KenBurnsMotion, MetricsTableScene, RepoMap, Storyboard,
    Submission, TextSpan, scene_id,
)


def minimal_submission() -> dict:
    return {
        "title": "Demo",
        "sources": [
            {"kind": "text", "id": "s1", "file": "README.md", "start_line": 1, "end_line": 1, "text": "# Demo"},
            {"kind": "json", "id": "s2", "file": "m.json", "pointer": "/f1", "value": 0.7349},
            {"kind": "image", "id": "s3", "file": "a.png"},
        ],
        "scenes": [
            {"type": "title", "heading": "Demo", "narration": [{"text": "Hi.", "source_ids": ["s1"]}]},
            {"type": "image", "image_id": "s3", "highlights": [[0.1, 0.1, 0.2, 0.2]],
             "motion": {"kind": "ken_burns", "from_rect": [0, 0, 1, 1], "to_rect": [0.2, 0.2, 0.5, 0.5]},
             "narration": [{"text": "Look.", "source_ids": ["s3"]}]},
            {"type": "metrics_table", "columns": ["Model", "F1"],
             "rows": [["small", {"span_id": "s2", "decimals": 3}]],
             "narration": [{"text": "Numbers.", "source_ids": ["s2"]}]},
            {"type": "outro", "repo_url": "https://github.com/o/r", "cta": "Try it",
             "narration": [{"text": "Bye.", "source_ids": ["s1"]}]},
        ],
    }


def test_submission_parses_discriminated_unions():
    sub = Submission.model_validate(minimal_submission())
    assert isinstance(sub.sources[0], TextSpan)
    assert isinstance(sub.scenes[1], ImageScene)
    assert isinstance(sub.scenes[1].motion, KenBurnsMotion)
    cell = sub.scenes[2].rows[0][1]
    assert isinstance(cell, CellRef) and cell.decimals == 3
    assert isinstance(sub.scenes[2], MetricsTableScene)


def test_unknown_scene_type_rejected():
    raw = minimal_submission()
    raw["scenes"][0]["type"] = "video"
    with pytest.raises(ValidationError):
        Submission.model_validate(raw)


def test_extra_fields_rejected():
    raw = minimal_submission()
    raw["scenes"][0]["html"] = "<script>"
    with pytest.raises(ValidationError):
        Submission.model_validate(raw)


def test_storyboard_round_trip_to_submission():
    sub = Submission.model_validate(minimal_submission())
    sb = Storyboard(repo_url="https://github.com/o/r", commit_sha="abc", title=sub.title,
                    sources={s.id: s for s in sub.sources}, scenes=sub.scenes,
                    llm={"provider": "fake", "model": "m"})
    again = Submission.model_validate(sb.to_submission())
    assert again == sub
    assert Storyboard.model_validate_json(sb.model_dump_json()) == sb


def test_scene_id_is_one_based_and_padded():
    assert scene_id(0) == "scene01"
    assert scene_id(11) == "scene12"


def test_repomap_file_lookup():
    rm = RepoMap(repo_url="u", commit_sha="c", root="/x", n_files=1,
                 files=[FileEntry(path="a.py", size=3, kind="text")], skipped=[], tree_text="")
    assert rm.file("a.py").kind == "text"
    assert rm.file("b.py") is None
