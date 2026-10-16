import json

import pytest

from codecast.agent.submit import Submitter
from codecast.errors import ValidationFailed
from codecast.grounding.validator import ValidationContext

from tests.unit.storyboards import valid_submission


def submitter(repo):
    return Submitter(ValidationContext(repomap=repo, llm={"provider": "fake", "model": "m"}, output_allowlist=["*.log"]))


def bad():
    raw = valid_submission()
    raw["scenes"][3]["narration"][0]["text"] = "The large@8 model reaches an F1 score of 0.74 overall."
    return raw


def test_accepts_valid(tiny_repo):
    s = submitter(tiny_repo)
    out = s(valid_submission())
    assert out.accepted and out.feedback == "accepted" and s.accepted is out.storyboard


def test_rejection_feedback_lists_errors_and_rounds_left(tiny_repo):
    s = submitter(tiny_repo)
    out = s(bad())
    data = json.loads(out.feedback)
    assert not out.accepted and data["rounds_left"] == 3
    assert data["errors"][0]["code"] == "UNGROUNDED_NUMBER"
    assert s.failures_by_code["UNGROUNDED_NUMBER"] == 1


def test_fourth_rejection_raises(tiny_repo):
    s = submitter(tiny_repo)
    for _ in range(3):
        s(bad())
    with pytest.raises(ValidationFailed) as ei:
        s(bad())
    assert ei.value.exit_code == 5 and ei.value.details["issues"]


def test_revision_gets_fresh_rounds_but_total_counts(tiny_repo):
    s = submitter(tiny_repo)
    s(bad()); s(bad()); s(valid_submission())
    s.begin_revision()
    s(bad()); s(bad()); s(bad())
    assert s(valid_submission()).accepted
    assert s.rounds_total == 7


def _by_reference():
    raw = valid_submission()
    for src in raw["sources"]:
        src.pop("text", None)
        src.pop("value", None)
    return raw


def test_by_reference_sources_are_filled_and_round_trip(tiny_repo):
    s = submitter(tiny_repo)
    out = s(_by_reference())
    assert out.accepted
    full = valid_submission()
    for src in full["sources"]:
        got = out.storyboard.sources[src["id"]]
        if src["kind"] == "text":
            assert got.text == src["text"]
        elif src["kind"] == "json":
            assert got.value == src["value"]
    again = submitter(tiny_repo)(out.storyboard.to_submission())
    assert again.accepted


def test_unchanged_narration_rejected_on_revision(tiny_repo):
    s = submitter(tiny_repo)
    first = s(valid_submission()).storyboard
    s.begin_revision(first)
    out = s(valid_submission())
    assert not out.accepted and "NARRATION_UNCHANGED" in out.feedback and "155" in out.feedback
    changed = valid_submission()
    changed["scenes"][2]["narration"][0]["text"] += " Really quite tiny."
    assert s(changed).accepted


def test_word_count_message_has_target(tiny_repo):
    raw = valid_submission()
    raw["scenes"][5]["narration"][0]["text"] = "Short outro."
    raw["scenes"][1]["narration"] = raw["scenes"][1]["narration"][:1]
    out = submitter(tiny_repo)(raw)
    assert "WORD_COUNT" in out.feedback and "155" in out.feedback


def test_text_span_with_trailing_newline_stored_canonical(tiny_repo):
    raw = valid_submission()
    raw["sources"][1]["text"] += "\n"
    out = submitter(tiny_repo)(raw)
    assert out.accepted and out.storyboard.sources["code"].text == 'def main():\n    print("hello")'


def test_nan_inf_and_long_decimals_unrepresentable():
    from decimal import Decimal
    from codecast.grounding.spans import _json_safe
    for bad_value in (Decimal("NaN"), Decimal("Infinity"), Decimal("0.1234567890123456789012")):
        with pytest.raises(ValueError):
            _json_safe(bad_value)
    assert _json_safe(Decimal("0.7349")) == 0.7349 and _json_safe(Decimal("12")) == 12


def test_accepted_storyboard_survives_disk_round_trip_and_revalidates(tiny_repo, tmp_path):
    from codecast.models import Storyboard
    sb = submitter(tiny_repo)(_by_reference()).storyboard
    path = tmp_path / "storyboard.json"
    path.write_text(sb.model_dump_json(indent=1))
    loaded = Storyboard.model_validate_json(path.read_text())
    assert submitter(tiny_repo)(loaded.to_submission()).accepted


def test_revision_message_uses_feedback_target(tiny_repo):
    s = submitter(tiny_repo)
    first = s(valid_submission()).storyboard
    s.begin_revision(first, target_words=118)
    assert "~118 words" in s(valid_submission()).feedback
