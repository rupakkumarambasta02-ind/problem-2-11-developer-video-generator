"""Walk a checkout into a RepoMap (spec §3)."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import pathspec

from codecast.ingest.sandbox import Sandbox, SandboxError
from codecast.models import FileEntry, RepoMap, SkippedEntry

IGNORED_DIRS = frozenset({"node_modules", ".venv", "venv", "vendor", "dist", "build", ".git", "__pycache__"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def _gitignore(root: Path) -> pathspec.PathSpec:
    gi = root / ".gitignore"
    lines = gi.read_text(encoding="utf-8", errors="replace").splitlines() if gi.is_file() else []
    return pathspec.GitIgnoreSpec.from_lines(lines)


def _join(base: str, name: str) -> str:
    return name if base in ("", ".") else f"{base}/{name}"


def build_repomap(root: Path, repo_url: str, commit_sha: str, *, max_file_bytes: int = 1_000_000) -> RepoMap:
    sandbox = Sandbox(root)
    root = sandbox.root
    spec = _gitignore(root)
    files: list[FileEntry] = []
    skipped: list[SkippedEntry] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        rel_dir = here.relative_to(root).as_posix()
        keep = []
        for name in sorted(dirnames):
            rel = _join(rel_dir, name)
            if (here / name).is_symlink():
                try:
                    sandbox.resolve(rel)
                except SandboxError:
                    skipped.append(SkippedEntry(path=rel, reason="symlink_escape"))
                continue
            if name in IGNORED_DIRS:
                skipped.append(SkippedEntry(path=rel, reason="ignored_dir"))
            elif spec.match_file(rel + "/"):
                skipped.append(SkippedEntry(path=rel, reason="gitignored"))
            else:
                keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            rel = _join(rel_dir, name)
            full = here / name
            if full.is_symlink():
                try:
                    sandbox.resolve(rel)
                    full.stat()
                except (SandboxError, OSError):
                    skipped.append(SkippedEntry(path=rel, reason="symlink_escape"))
                    continue
            if spec.match_file(rel):
                skipped.append(SkippedEntry(path=rel, reason="gitignored"))
                continue
            size = full.stat().st_size
            if size > max_file_bytes:
                skipped.append(SkippedEntry(path=rel, reason="too_large"))
                continue
            with open(full, "rb") as fh:
                head = fh.read(8192)
            if head.startswith(LFS_HEADER):
                skipped.append(SkippedEntry(path=rel, reason="lfs_pointer"))
                continue
            if full.suffix.lower() in IMAGE_SUFFIXES:
                kind = "image"
            elif b"\x00" in head:
                kind = "binary"
            else:
                kind = "text"
            files.append(FileEntry(path=rel, size=size, kind=kind))
    files.sort(key=lambda f: f.path)
    tree = f"{len(files)} files, {len(skipped)} skipped\n" + render_tree(files)
    return RepoMap(repo_url=repo_url, commit_sha=commit_sha, root=str(root), n_files=len(files),
                   files=files, skipped=skipped, tree_text=tree)


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024 or unit == "MB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def render_tree(files: list[FileEntry], *, max_depth: int = 3, max_lines: int = 250) -> str:
    children: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    sizes = {f.path: f.size for f in files}
    for f in files:
        parts = f.path.split("/")
        for i in range(len(parts)):
            parent = "/".join(parts[:i])
            children[parent].add("/".join(parts[: i + 1]))
            if i < len(parts) - 1:
                counts["/".join(parts[: i + 1])] += 1
    lines: list[str] = []

    def walk(node: str, depth: int) -> None:
        for child in sorted(children.get(node, ()), key=lambda c: (c not in counts, c)):
            name = child.rsplit("/", 1)[-1]
            indent = "  " * depth
            if child in counts:
                lines.append(f"{indent}{name}/ ({counts[child]} files)")
                if depth + 1 < max_depth:
                    walk(child, depth + 1)
            else:
                lines.append(f"{indent}{name} ({_human(sizes[child])})")

    walk("", 0)
    if len(lines) > max_lines:
        extra = len(lines) - max_lines
        lines = lines[:max_lines] + [f"... ({extra} more entries)"]
    return "\n".join(lines)
