import subprocess

import pytest

from codecast.errors import IngestError
from codecast.ingest import clone as clone_mod
from codecast.ingest.clone import ingest, normalize_repo_url, repo_name


@pytest.mark.parametrize("url", [
    "https://github.com/o/r", "https://github.com/o/r.git", "https://github.com/o/r/",
    " https://github.com/o/r.git/ ",
])
def test_normalize_variants(url):
    assert normalize_repo_url(url) == "https://github.com/o/r"


@pytest.mark.parametrize("url", ["git@github.com:o/r.git", "http://github.com/o/r", "ftp://x/y", "o/r"])
def test_rejects_non_https(url):
    with pytest.raises(IngestError, match="https"):
        normalize_repo_url(url)


def test_repo_name():
    assert repo_name("https://github.com/o/video-tool") == "video-tool"
    assert repo_name("/tmp/some/dir/") == "dir"


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                   capture_output=True)


@pytest.fixture
def local_repo(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    git(src, "init", "-q", "-b", "main")
    (src / "README.md").write_text("# hi\n")
    git(src, "add", ".")
    git(src, "commit", "-q", "-m", "init")
    (src / "uncommitted.txt").write_text("not in HEAD\n")
    git(src, "remote", "add", "origin", "https://github.com/o/r.git")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()
    return src, sha


def test_ingest_local_path(local_repo, tmp_path):
    src, sha = local_repo
    rm = ingest(str(src), tmp_path / "ws" / "repo")
    assert rm.commit_sha == sha
    assert rm.repo_url == "https://github.com/o/r"
    paths = {f.path for f in rm.files}
    assert "README.md" in paths and "uncommitted.txt" not in paths


def test_ingest_size_cap(local_repo, tmp_path):
    src, _ = local_repo
    with pytest.raises(IngestError, match="exceeds"):
        ingest(str(src), tmp_path / "ws" / "repo", max_bytes=1)


def test_ingest_timeout(monkeypatch, tmp_path):
    def boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="git", timeout=120)
    monkeypatch.setattr(clone_mod.subprocess, "run", boom)
    with pytest.raises(IngestError, match="timed out"):
        ingest("https://github.com/o/r", tmp_path / "repo")


def test_ingest_clone_failure_has_stderr(monkeypatch, tmp_path):
    def fail(*a, **kw):
        raise subprocess.CalledProcessError(128, "git", stderr="fatal: repository not found")
    monkeypatch.setattr(clone_mod.subprocess, "run", fail)
    with pytest.raises(IngestError, match="repository not found"):
        ingest("https://github.com/o/missing", tmp_path / "repo")


def test_clone_env_never_prompts(monkeypatch, tmp_path):
    seen = {}

    def capture(cmd, **kw):
        seen.update(kw["env"])
        raise subprocess.CalledProcessError(128, cmd, stderr="x")
    monkeypatch.setattr(clone_mod.subprocess, "run", capture)
    with pytest.raises(IngestError):
        ingest("https://github.com/o/r", tmp_path / "repo")
    assert seen["GIT_TERMINAL_PROMPT"] == "0" and seen["GIT_LFS_SKIP_SMUDGE"] == "1"
