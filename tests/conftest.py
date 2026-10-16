from pathlib import Path

import pytest
from PIL import Image

from codecast.models import FileEntry, RepoMap


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    """Offline tests must never see real API keys."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


README = (
    "# Tiny\n"
    "Tiny classifies clips with ViT-B-32 at 0.7349 F1.\n"
    "Run `python app.py --top 3` to predict.\n"
)
METRICS = ('{"models": {"small": {"f1": 0.677, "latency_ms": 420}, '
           '"large@8": {"f1": 0.7349, "latency_ms": 891.2}}, "n_videos": 90}\n')
APP = 'def main():\n    print("hello")\n'


@pytest.fixture
def tiny_repo(tmp_path: Path) -> RepoMap:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "README.md").write_text(README)
    (root / "metrics.json").write_text(METRICS)
    (root / "src" / "app.py").write_text(APP)
    (root / "logs" / "run.log").write_text("accuracy 0.91\n")
    (root / "crlf.txt").write_bytes(b"\xef\xbb\xbfalpha 1\r\nbeta 2\r\n")
    (root / "bad.txt").write_bytes(b"ok\n\xff\xfe broken\n")
    Image.new("RGB", (400, 300), (30, 60, 90)).save(root / "plot.png")
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            files.append(FileEntry(path=rel, size=p.stat().st_size,
                                   kind="image" if p.suffix == ".png" else "text"))
    return RepoMap(repo_url="https://github.com/o/tiny", commit_sha="c0ffee", root=str(root),
                   n_files=len(files), files=files, skipped=[], tree_text="")
