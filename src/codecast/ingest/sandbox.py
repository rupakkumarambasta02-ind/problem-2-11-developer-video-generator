"""Every agent-supplied path is resolved with realpath and must stay inside the repo root."""

from __future__ import annotations

import os
from pathlib import Path


class SandboxError(Exception):
    pass


class Sandbox:
    def __init__(self, root: Path):
        self.root = Path(os.path.realpath(root))

    def resolve(self, rel: str) -> Path:
        if not isinstance(rel, str) or "\x00" in rel:
            raise SandboxError("invalid path")
        candidate = Path(os.path.realpath(self.root / rel))
        if candidate != self.root and self.root not in candidate.parents:
            raise SandboxError(f"path escapes the repository: {rel}")
        return candidate

    def relative(self, path: Path) -> str:
        rel = Path(os.path.realpath(path)).relative_to(self.root).as_posix()
        return rel or "."
