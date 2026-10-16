import json
import shutil
import subprocess
from pathlib import Path

import pytest

from codecast.config import load_config
from codecast.llm.fake import FakeProvider, tool_call
from codecast.pipeline import RunOptions, run

from tests.fakes import SilenceEngine
from tests.unit.storyboards import valid_submission

pytestmark = pytest.mark.integration


@pytest.fixture
def git_repo(tiny_repo, tmp_path):
    src = Path(tiny_repo.root)
    for args in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-q", "-m", "init"],
                 ["remote", "add", "origin", "https://github.com/o/tiny"]):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=src, check=True,
                       capture_output=True)
    return src


def opts(git_repo, tmp_path, **kw):
    return RunOptions(repo=str(git_repo), out_dir=tmp_path / "output", workspace=tmp_path / "ws", run_id="r1", **kw)


def submit():
    return tool_call("submit_storyboard", {"storyboard": valid_submission()})


def submit_variant(n):
    """Same storyboard with revised narration (a duration revision must change the words)."""
    raw = valid_submission()
    raw["scenes"][2]["narration"][0]["text"] += " " + "Truly. " * n
    return tool_call("submit_storyboard", {"storyboard": raw})


def test_happy_path_produces_all_artifacts(git_repo, tmp_path):
    # 130 words * 0.4 s = 52 s speech + 2 gaps + 6 tails ≈ 55.4 s → inside [50, 80]
    rep = run(opts(git_repo, tmp_path), load_config(), provider=FakeProvider([submit()]),
              engine=SilenceEngine(0.4), preflight=False)
    assert rep.status == "ok", rep.failure
    out = tmp_path / "output" / "tiny"
    for name in ("explainer.mp4", "explainer.srt", "storyboard.json", "run_report.json"):
        assert (out / name).is_file()
    assert 50 <= rep.video.duration_s <= 80
    assert rep.validation.storyboard_rounds == 1 and rep.tts_engine == "silence"
    assert set(rep.stage_seconds) == {"ingest", "agent", "voice", "render", "compose"}
    assert json.loads((out / "storyboard.json").read_text())["repo_url"] == "https://github.com/o/tiny"
    assert not list(out.glob(".staging-*"))


def test_duration_loop_revises_then_succeeds(git_repo, tmp_path):
    engine = SilenceEngine(0.7)                       # first pass ≈ 95 s → too long
    provider = FakeProvider([submit(), lambda req: _shorter(engine) or submit_variant(1)])
    rep = run(opts(git_repo, tmp_path), load_config(), provider=provider, engine=engine, preflight=False)
    assert rep.status == "ok", rep.failure
    assert rep.validation.duration_rounds == 1
    assert '"target": [50.0, 80.0]' in provider.requests[1].messages[-1].text()


def _shorter(engine):
    engine.seconds_per_word = 0.4                     # the "revised" narration speaks faster
    return None


def test_duration_failure_exit_3_and_existing_output_untouched(git_repo, tmp_path):
    out = tmp_path / "output" / "tiny"
    out.mkdir(parents=True)
    (out / "explainer.mp4").write_bytes(b"committed demo")
    rep = run(opts(git_repo, tmp_path), load_config(), provider=FakeProvider([submit(), submit_variant(1), submit_variant(2)]),
              engine=SilenceEngine(0.7), preflight=False)
    assert rep.exit_code == 3 and rep.validation.duration_rounds == 2
    assert (out / "explainer.mp4").read_bytes() == b"committed demo"
    assert json.loads((out / "last_failed_run_report.json").read_text())["exit_code"] == 3
    assert (tmp_path / "ws" / "r1" / "run_report.json").is_file()


def test_ingest_failure_exit_2(tmp_path):
    rep = run(RunOptions(repo="git@github.com:o/r.git", out_dir=tmp_path / "o", workspace=tmp_path / "ws", run_id="r2"),
              load_config(), provider=FakeProvider([]), engine=SilenceEngine(), preflight=False)
    assert rep.exit_code == 2 and rep.failure.stage == "ingest"


def test_resume_from_render_reuses_workspace(git_repo, tmp_path):
    first = run(opts(git_repo, tmp_path), load_config(), provider=FakeProvider([submit()]),
                engine=SilenceEngine(0.4), preflight=False)
    assert first.status == "ok"
    shutil.rmtree(tmp_path / "output")
    rep = run(opts(git_repo, tmp_path, from_stage="render"), load_config(), provider=FakeProvider([]),
              engine=SilenceEngine(0.4), preflight=False)
    assert rep.status == "ok", rep.failure
    assert (tmp_path / "output" / "tiny" / "explainer.mp4").is_file()


def test_resume_from_voice_out_of_range_exit_3_output_untouched(git_repo, tmp_path):
    first = run(opts(git_repo, tmp_path), load_config(), provider=FakeProvider([submit()]),
                engine=SilenceEngine(0.4), preflight=False)
    assert first.status == "ok"
    mp4 = tmp_path / "output" / "tiny" / "explainer.mp4"
    before = mp4.read_bytes()
    shutil.rmtree(tmp_path / "ws" / "r1" / "audio")      # the audio cache would otherwise be reused as-is
    rep = run(opts(git_repo, tmp_path, from_stage="voice"), load_config(), provider=FakeProvider([]),
              engine=SilenceEngine(0.7), preflight=False)
    assert rep.exit_code == 3 and rep.failure.stage == "voice"
    assert mp4.read_bytes() == before


def test_agent_failure_records_stats(git_repo, tmp_path):
    from codecast.llm.fake import say
    rep = run(opts(git_repo, tmp_path), load_config(), provider=FakeProvider([say("a"), say("b"), say("c")]),
              engine=SilenceEngine(0.4), preflight=False)
    assert rep.exit_code == 4 and rep.failure.stage == "agent"
    assert rep.budget.steps_used == 3


def test_invalid_from_stage_returns_report_exit_7(git_repo, tmp_path):
    rep = run(RunOptions(repo=str(git_repo), out_dir=tmp_path / "o", workspace=tmp_path / "ws", from_stage="render"),
              load_config(), provider=FakeProvider([]), engine=SilenceEngine(0.4), preflight=False)
    assert rep.exit_code == 7


def test_unwritable_out_dir_still_returns_report(git_repo, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    rep = run(RunOptions(repo="git@github.com:o/r.git", out_dir=blocker / "o", workspace=tmp_path / "ws", run_id="r9"),
              load_config(), provider=FakeProvider([]), engine=SilenceEngine(), preflight=False)
    assert rep.exit_code == 2


def test_resume_carries_agent_accounting_forward(git_repo, tmp_path):
    from codecast.llm.types import Usage
    first = run(opts(git_repo, tmp_path), load_config(),
                provider=FakeProvider([tool_call("submit_storyboard", {"storyboard": valid_submission()},
                                                 usage=Usage(1000, 200, 300, 0))]),
                engine=SilenceEngine(0.4), preflight=False)
    assert first.status == "ok" and first.usage.input == 1000
    rep = run(opts(git_repo, tmp_path, from_stage="render"), load_config(), provider=FakeProvider([]),
              engine=SilenceEngine(0.4), preflight=False)
    assert rep.status == "ok", rep.failure
    assert rep.usage == first.usage and rep.budget == first.budget and rep.validation == first.validation
    assert rep.cost_usd == first.cost_usd and rep.cost_unknown_reason == first.cost_unknown_reason
    assert rep.llm == first.llm and rep.commit_sha == first.commit_sha
    for st in ("ingest", "agent", "voice"):
        assert rep.stage_seconds[st] == first.stage_seconds[st]
    assert set(rep.stage_seconds) == {"ingest", "agent", "voice", "render", "compose"}


def test_resume_without_prior_report_still_works(git_repo, tmp_path):
    run(opts(git_repo, tmp_path), load_config(), provider=FakeProvider([submit()]),
        engine=SilenceEngine(0.4), preflight=False)
    (tmp_path / "ws" / "r1" / "run_report.json").unlink()
    rep = run(opts(git_repo, tmp_path, from_stage="render"), load_config(), provider=FakeProvider([]),
              engine=SilenceEngine(0.4), preflight=False)
    assert rep.status == "ok", rep.failure
    assert set(rep.stage_seconds) == {"render", "compose"}


def test_resume_with_unparsable_prior_report_still_works(git_repo, tmp_path):
    run(opts(git_repo, tmp_path), load_config(), provider=FakeProvider([submit()]),
        engine=SilenceEngine(0.4), preflight=False)
    (tmp_path / "ws" / "r1" / "run_report.json").write_text("{not json")
    rep = run(opts(git_repo, tmp_path, from_stage="render"), load_config(), provider=FakeProvider([]),
              engine=SilenceEngine(0.4), preflight=False)
    assert rep.status == "ok", rep.failure
