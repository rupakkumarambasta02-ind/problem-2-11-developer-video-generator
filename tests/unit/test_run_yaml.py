from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_run_yaml_shape():
    data = yaml.safe_load((ROOT / "run.yaml").read_text())
    assert set(data) == {"setup_commands", "run_commands", "test_commands", "notes"}
    setup = " ".join(data["setup_commands"])
    for needle in ("sys.version_info >= (3, 12)", "pip install -e '.[dev]'", "playwright install chromium",
                   "codecast fetch-voice", "codecast doctor"):
        assert needle in setup
    run = " ".join(data["run_commands"])
    assert "codecast run --repo https://github.com/rkumar-bengaluru/video-classifier --out output/" in run
    assert "output/video-classifier/explainer.mp4" in run
    assert data["test_commands"] == [".venv/bin/python -m pytest -q"]
    assert not any("||" in c or "&&" in c for c in data["setup_commands"] + data["run_commands"])


def test_committed_demo_paths_are_documented():
    readme = (ROOT / "README.md").read_text()
    assert "output/video-classifier/explainer.mp4" in readme
    assert "ANTHROPIC_API_KEY" in readme and "OPENAI_API_KEY" in readme
    current = readme.split("## 9. What changed")[0]          # the corrections table may name the old tools
    assert "CLAUDE_KEY" not in current and "mmdc" not in current
