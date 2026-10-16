import os

import pytest

from codecast.ingest.sandbox import Sandbox, SandboxError


@pytest.fixture
def box(tmp_path):
    root = tmp_path / "repo"
    (root / "a").mkdir(parents=True)
    (root / "a" / "f.txt").write_text("x")
    (tmp_path / "secret.txt").write_text("s")
    os.symlink(tmp_path / "secret.txt", root / "escape.txt")
    os.symlink(root / "a" / "f.txt", root / "inside.txt")
    return Sandbox(root)


def test_resolves_inside(box):
    assert box.resolve("a/f.txt").read_text() == "x"
    assert box.resolve(".") == box.root
    assert box.resolve("") == box.root
    assert box.resolve("inside.txt").name == "f.txt"


@pytest.mark.parametrize("rel", ["../secret.txt", "a/../../secret.txt", "escape.txt", "/etc/passwd", "a/\x00"])
def test_rejects_escape(box, rel):
    with pytest.raises(SandboxError):
        box.resolve(rel)


def test_relative(box):
    assert box.relative(box.root / "a" / "f.txt") == "a/f.txt"
    assert box.relative(box.root) == "."
