import json
from pathlib import Path

import pytest

from codecast.models import Storyboard, Submission
from codecast.voice.timeline import (
    build_timeline, duration_feedback, split_cues, srt_timestamp, synthesize_storyboard, write_srt,
)

from tests.fakes import SilenceEngine
from tests.unit.storyboards import valid_submission


def test_build_timeline_gaps_tails_and_frame_rounding():
    tl = build_timeline([("scene01", [("a", Path("a.wav"), 1.0), ("b", Path("b.wav"), 2.0)]),
                         ("scene02", [("c", Path("c.wav"), 1.01)])], gap_s=0.15, tail_s=0.5)
    s1, s2 = tl.scenes
    assert s1.sentences[1].start == pytest.approx(1.15)
    assert s1.duration == pytest.approx(3.65 + 1 / 60, abs=1 / 30)   # ceil to frame
    assert round(s1.duration * 30, 6) == int(round(s1.duration * 30))
    assert s2.start == pytest.approx(s1.duration)
    assert s2.cues[0].start == pytest.approx(s2.start)
    assert tl.total == pytest.approx(s1.duration + s2.duration)
    assert tl.per_scene_seconds() == [s1.duration, s2.duration]
    assert tl.narration_seconds == pytest.approx(4.01)


def test_split_cues_short_and_long():
    assert [c.text for c in split_cues("short", 1.0, 2.0)] == ["short"]
    long = " ".join(["word"] * 40)          # 199 chars
    cues = split_cues(long, 10.0, 6.0)
    assert len(cues) == 3 and all(len(c.text) <= 84 for c in cues)
    assert cues[0].start == 10.0 and cues[-1].end == pytest.approx(16.0)
    assert all(a.end == pytest.approx(b.start) for a, b in zip(cues, cues[1:]))


def test_srt(tmp_path):
    assert srt_timestamp(3725.5) == "01:02:05,500"
    tl = build_timeline([("scene01", [("Hello there.", Path("a.wav"), 1.0)])], gap_s=0.15, tail_s=0.5)
    write_srt(tl, tmp_path / "x.srt")
    assert (tmp_path / "x.srt").read_text() == "1\n00:00:00,000 --> 00:00:01,000\nHello there.\n"


def storyboard():
    sub = Submission.model_validate(valid_submission())
    return Storyboard(repo_url="u", commit_sha="c", title=sub.title, sources={s.id: s for s in sub.sources},
                      scenes=sub.scenes)


def test_synthesize_storyboard_caches(tmp_path):
    eng = SilenceEngine(0.4)
    sb = storyboard()
    tl = synthesize_storyboard(sb, eng, tmp_path, gap_s=0.15, tail_s=0.5)
    assert [s.scene_id for s in tl.scenes] == [f"scene0{i}" for i in range(1, 7)]
    n = len(eng.calls)
    synthesize_storyboard(sb, eng, tmp_path, gap_s=0.15, tail_s=0.5)
    assert len(eng.calls) == n          # all cached
    assert tl.narration_seconds == pytest.approx(130 * 0.4, rel=0.01)


def test_duration_feedback(tmp_path):
    sb = storyboard()
    tl = synthesize_storyboard(sb, SilenceEngine(0.5), tmp_path, gap_s=0.15, tail_s=0.5)
    data = json.loads(duration_feedback(tl, sb))
    assert data["target"] == [50.0, 80.0] and data["total"] == pytest.approx(tl.total)
    assert len(data["per_scene_seconds"]) == 6
    assert "shorten" in data["instruction"]


def test_duration_feedback_gives_a_target_word_count(tmp_path):
    sb = storyboard()
    tl = synthesize_storyboard(sb, SilenceEngine(0.45), tmp_path, gap_s=0.15, tail_s=0.5)
    data = json.loads(duration_feedback(tl, sb))
    words = data["spoken_words"]
    wps = words / tl.narration_seconds
    target_words = words + round((65.0 - tl.total) * wps)
    assert data["target_words"] == target_words
    assert f"from {words} to about {target_words} words" in data["instruction"]
    assert "unchanged narration will fail again" in data["instruction"]


def test_duration_feedback_never_targets_below_the_word_minimum(tmp_path):
    sb = storyboard()  # 130 words; at 0.9 s/word the naive target would be far below 110
    tl = synthesize_storyboard(sb, SilenceEngine(0.9), tmp_path, gap_s=0.15, tail_s=0.5)
    data = json.loads(duration_feedback(tl, sb))
    assert data["target_words"] == 110
    assert "from 130 to about 110 words" in data["instruction"]
    assert "Numbers, identifiers" in data["instruction"]
