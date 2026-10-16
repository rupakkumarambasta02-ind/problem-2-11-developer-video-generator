import os

from PIL import Image

from codecast.ingest.repomap import LFS_HEADER, build_repomap


def make_repo(tmp_path):
    root = tmp_path / "repo"
    for d in ("src", "node_modules/pkg", "build", "data", "docs"):
        (root / d).mkdir(parents=True)
    (root / "README.md").write_text("# R\n")
    (root / "src" / "main.py").write_text("print(1)\n")
    (root / "node_modules" / "pkg" / "index.js").write_text("x")
    (root / "build" / "out.txt").write_text("x")
    (root / ".gitignore").write_text("*.secret\ndata/\n")
    (root / "keys.secret").write_text("x")
    (root / "data" / "big.csv").write_text("x")
    (root / "huge.txt").write_bytes(b"a" * 1_000_001)
    (root / "model.bin").write_bytes(LFS_HEADER + b"\noid sha256:abc\nsize 12345\n")
    (root / "blob.dat").write_bytes(b"\x00\x01\x02")
    Image.new("RGB", (10, 10)).save(root / "docs" / "plot.png")
    (tmp_path / "outside.txt").write_text("x")
    os.symlink(tmp_path / "outside.txt", root / "link.txt")
    return root


def test_build_repomap_skip_rules(tmp_path):
    root = make_repo(tmp_path)
    rm = build_repomap(root, "https://github.com/o/r", "abc123")
    kinds = {f.path: f.kind for f in rm.files}
    assert kinds["README.md"] == "text"
    assert kinds["src/main.py"] == "text"
    assert kinds["docs/plot.png"] == "image"
    assert kinds["blob.dat"] == "binary"
    assert kinds[".gitignore"] == "text"
    reasons = {s.path: s.reason for s in rm.skipped}
    assert reasons["node_modules"] == "ignored_dir"
    assert reasons["build"] == "ignored_dir"
    assert reasons["keys.secret"] == "gitignored"
    assert reasons["data"] == "gitignored"
    assert reasons["huge.txt"] == "too_large"
    assert reasons["model.bin"] == "lfs_pointer"
    assert reasons["link.txt"] == "symlink_escape"
    assert rm.n_files == len(rm.files)
    assert rm.commit_sha == "abc123"
    assert rm.root == str(root.resolve())


def test_tree_text_lists_files_with_sizes(tmp_path):
    rm = build_repomap(make_repo(tmp_path), "u", "c")
    assert "README.md" in rm.tree_text
    assert "src/" in rm.tree_text and "main.py" in rm.tree_text
    assert "node_modules" not in rm.tree_text


def test_tree_text_depth_and_line_limits(tmp_path):
    root = tmp_path / "deep"
    (root / "a" / "b" / "c" / "d").mkdir(parents=True)
    (root / "a" / "b" / "c" / "d" / "x.txt").write_text("x")
    for i in range(400):
        (root / f"f{i:03d}.txt").write_text("x")
    rm = build_repomap(root, "u", "c")
    assert "x.txt" not in rm.tree_text           # deeper than max_depth
    assert "c/ (1 files)" in rm.tree_text
    assert "more entries" in rm.tree_text         # line cap reached
    assert len(rm.tree_text.splitlines()) <= 252
