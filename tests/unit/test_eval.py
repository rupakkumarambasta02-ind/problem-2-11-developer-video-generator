import json
import subprocess
from pathlib import Path

import pytest

from codecast import eval as ev
from codecast.config import load_config
from codecast.models import Storyboard, Submission
from codecast.report import ReportBuilder, VideoInfo
from codecast.voice.timeline import srt_timestamp

from tests.unit.storyboards import valid_submission


@pytest.fixture
def committed(tiny_repo):
    root = Path(tiny_repo.root)
    for args in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True, capture_output=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    return root, sha


def write_outputs(out: Path, sha: str, raw: dict, narration_s: float = 52.0):
    out.mkdir(parents=True)
    sub = Submission.model_validate(raw)
    sb = Storyboard(repo_url="https://github.com/o/tiny", commit_sha=sha, title=sub.title,
                    sources={s.id: s for s in sub.sources}, scenes=sub.scenes, llm={"provider": "x", "model": "m"})
    (out / "storyboard.json").write_text(sb.model_dump_json())
    rb = ReportBuilder("r", sb.repo_url, load_config())
    rb.report.cost_usd = 0.04
    rb.report.stage_seconds = {"ingest": 3, "agent": 80, "voice": 20, "render": 15, "compose": 30}
    rb.report.video = VideoInfo(path=str(out / "explainer.mp4"), duration_s=55.0, size_bytes=1, narration_s=narration_s)
    (out / "run_report.json").write_text(rb.finish().model_dump_json())
    (out / "explainer.srt").write_text(f"1\n{srt_timestamp(0)} --> {srt_timestamp(narration_s)}\nall\n")
    (out / "explainer.mp4").write_bytes(b"x")


def fake_probe(monkeypatch, seconds=55.0):
    monkeypatch.setattr(ev, "probe", lambda p: {"format": {"duration": str(seconds)},
                                                "streams": [{"codec_type": "video", "duration": str(seconds)},
                                                            {"codec_type": "audio", "duration": str(seconds)}]})


def test_eval_passes_and_appends_log(committed, tmp_path, monkeypatch, capsys):
    root, sha = committed
    fake_probe(monkeypatch)
    out = tmp_path / "output" / "tiny"
    write_outputs(out, sha, valid_submission())
    log = tmp_path / "eval.md"
    assert ev.evaluate(out, source=str(root), eval_log=log) == 0
    printed = capsys.readouterr().out
    assert "| Ungrounded numbers / identifiers | 0 | 0 | ✓ |" in printed
    assert "55.0 s" in printed and "$0.0400" in printed and "2.5 min" in printed
    assert log.read_text().startswith("## ") and sha[:7] in log.read_text()


def test_eval_recheck_catches_tampered_storyboard(committed, tmp_path, monkeypatch):
    root, sha = committed
    fake_probe(monkeypatch)
    raw = valid_submission()
    raw["scenes"][3]["narration"][0]["text"] = "The large@8 model reaches an F1 score of 0.99 overall."
    out = tmp_path / "output" / "tiny"
    write_outputs(out, sha, raw)
    assert ev.evaluate(out, source=str(root), eval_log=tmp_path / "e.md") == 1


def test_eval_flags_low_caption_coverage(committed, tmp_path, monkeypatch):
    root, sha = committed
    fake_probe(monkeypatch)
    out = tmp_path / "output" / "tiny"
    write_outputs(out, sha, valid_submission(), narration_s=52.0)
    (out / "explainer.srt").write_text(f"1\n{srt_timestamp(0)} --> {srt_timestamp(30)}\nhalf\n")
    assert ev.evaluate(out, source=str(root), eval_log=tmp_path / "e.md") == 1


def test_srt_union():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.srt"
        p.write_text("1\n00:00:00,000 --> 00:00:02,000\na\n\n2\n00:00:01,000 --> 00:00:03,000\nb\n")
        assert ev.srt_covered_seconds(p) == pytest.approx(3.0)
