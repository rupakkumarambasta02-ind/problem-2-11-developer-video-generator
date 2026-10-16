"""Real LLM + real repos. Excluded by default; run with: pytest -m live tests/e2e/test_live.py"""

import os
from pathlib import Path

import pytest

from codecast import eval as ev
from codecast.config import load_config
from codecast.pipeline import RunOptions, run

pytestmark = pytest.mark.live
CONFIG = os.environ.get("CODECAST_LIVE_CONFIG")


def cfg():
    return load_config(Path(CONFIG) if CONFIG else None)


def test_demo_repo(tmp_path):
    rep = run(RunOptions(repo="https://github.com/rkumar-bengaluru/video-classifier", out_dir=tmp_path / "out",
                         workspace=tmp_path / "ws"), cfg())
    assert rep.status == "ok", rep.failure
    assert 50 <= rep.video.duration_s <= 80
    assert ev.evaluate(tmp_path / "out" / "video-classifier", eval_log=tmp_path / "eval.md") == 0


def test_smoke_repo_without_metrics(tmp_path):
    rep = run(RunOptions(repo="https://github.com/pallets/markupsafe", out_dir=tmp_path / "out",
                         workspace=tmp_path / "ws"), cfg())
    assert rep.status == "ok", rep.failure
