"""Read-only, sandboxed tools for the director agent (spec §5.2). Repo content is untrusted."""

from __future__ import annotations

import fnmatch
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from codecast.errors import CodecastError
from codecast.grounding.spans import PointerError, dumps_decimal, load_data_decimal, read_lines, resolve_pointer
from codecast.ingest.sandbox import Sandbox, SandboxError
from codecast.llm.types import ImagePart, TextPart, ToolCallPart, ToolResultPart, ToolSpec
from codecast.models import RepoMap, Storyboard, Submission

MAX_RESULT_BYTES = 32 * 1024
IMAGE_LONG_EDGE = 1568


@dataclass
class SubmitOutcome:
    accepted: bool
    storyboard: Storyboard | None
    feedback: str


@dataclass
class ToolExecution:
    result: ToolResultPart
    storyboard: Storyboard | None = None


class ToolArgError(Exception):
    pass


# Serialized line budget for read_file; the rest of the envelope fits in the remaining 1 KiB.
_LINES_BUDGET = MAX_RESULT_BYTES - 1024
_TRUNC_MARK = " …[line truncated]"


def _json_size(text: str) -> int:
    """Bytes this string occupies as a JSON array element (quotes, escapes, ', ' separator)."""
    return len(json.dumps(text, ensure_ascii=False).encode()) + 2


def _fit_line(text: str, budget: int) -> str:
    """Shortest-prefix cut so the JSON-encoded line plus marker fits the budget."""
    if _json_size(text) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:  # largest prefix length whose encoded size fits
        mid = (lo + hi + 1) // 2
        if _json_size(text[:mid] + _TRUNC_MARK) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + _TRUNC_MARK


def _obj(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


_PATH = {"type": "string", "description": "Path relative to the repository root ('.' is the root)."}
_SPECS = {
    "list_dir": ToolSpec("list_dir", "List files and directories (with kind and size) under a repo directory.",
                         _obj({"path": _PATH, "depth": {"type": "integer", "minimum": 1, "maximum": 3}}, ["path"])),
    "read_file": ToolSpec("read_file", "Read a text file. Lines are returned as '<n>| <text>'; when citing, copy "
                          "<text> exactly, without the '<n>| ' prefix. Use next_start_line to page.",
                          _obj({"path": _PATH, "start_line": {"type": "integer", "minimum": 1},
                                "max_lines": {"type": "integer", "minimum": 1, "maximum": 400}}, ["path"])),
    "grep": ToolSpec("grep", "Search text files with a Python regular expression.",
                     _obj({"pattern": {"type": "string"}, "path_glob": {"type": "string"},
                           "max_results": {"type": "integer", "minimum": 1, "maximum": 200}}, ["pattern"])),
    "read_json": ToolSpec("read_json", "Read the value at an RFC 6901 JSON Pointer in a .json, .yml or .yaml file "
                                       "('' = whole document).",
                          _obj({"path": _PATH, "pointer": {"type": "string"}}, ["path"])),
    "view_image": ToolSpec("view_image", "Look at an image file from the repo.", _obj({"path": _PATH}, ["path"])),
}
_PY_TYPES = {"string": str, "integer": int}


def _submit_spec() -> ToolSpec:
    schema = Submission.model_json_schema()
    defs = schema.pop("$defs", {})
    return ToolSpec("submit_storyboard",
                    "Submit the complete storyboard. Returns 'accepted' or a list of validation errors to fix. "
                    "Cite text by file + start_line + end_line (omit `text`; the tool fills the exact lines). "
                    "Cite JSON or YAML data by file + pointer (omit `value`).",
                    {"type": "object", "properties": {"storyboard": schema}, "required": ["storyboard"],
                     "$defs": defs})


def _check_args(spec: ToolSpec, args: dict) -> None:
    if "__invalid_json__" in args:
        raise ToolArgError("tool arguments were not valid JSON; send a JSON object")
    props = spec.input_schema["properties"]
    for key in spec.input_schema.get("required", []):
        if key not in args:
            raise ToolArgError(f"missing required argument '{key}'")
    for key, value in args.items():
        if key not in props:
            raise ToolArgError(f"unexpected argument '{key}'; allowed: {sorted(props)}")
        typ = _PY_TYPES.get(props[key].get("type"))
        if typ is not None and (not isinstance(value, typ) or isinstance(value, bool)):
            raise ToolArgError(f"'{key}' must be {props[key]['type']}")
        if typ is int:
            if "minimum" in props[key] and value < props[key]["minimum"]:
                raise ToolArgError(f"'{key}' must be at least {props[key]['minimum']}")
            if "maximum" in props[key] and value > props[key]["maximum"]:
                raise ToolArgError(f"'{key}' must be at most {props[key]['maximum']}")


class ToolBox:
    def __init__(self, repomap: RepoMap, submit: Callable[[Any], SubmitOutcome]):
        self.repomap = repomap
        self.sandbox = Sandbox(Path(repomap.root))
        self._submit = submit
        self._files = {f.path: f for f in repomap.files}
        self._skipped = {s.path: s.reason for s in repomap.skipped}
        self._specs = {**_SPECS, "submit_storyboard": _submit_spec()}

    def specs(self, *, vision: bool) -> list[ToolSpec]:
        order = ["list_dir", "read_file", "grep", "read_json"] + (["view_image"] if vision else []) + ["submit_storyboard"]
        return [self._specs[n] for n in order]

    def execute(self, call: ToolCallPart) -> ToolExecution:
        spec = self._specs.get(call.name)
        try:
            if spec is None:
                raise ToolArgError(f"unknown tool '{call.name}'; available: {sorted(self._specs)}")
            _check_args(spec, call.args)
            return getattr(self, f"_t_{call.name}")(call.id, **call.args)
        except CodecastError:
            raise
        except (ToolArgError, SandboxError, PointerError, OSError, ValueError, RecursionError,
                Image.DecompressionBombError, re.error) as e:
            msg = f"invalid regex: {e}" if isinstance(e, re.error) else str(e)
            return ToolExecution(ToolResultPart(call.id, [TextPart(msg)], is_error=True))

    # ---- helpers ----
    def _ok(self, call_id: str, text: str) -> ToolExecution:
        return ToolExecution(ToolResultPart(call_id, [TextPart(text)]))

    def _entry(self, path: str, want: str):
        rel = self.sandbox.relative(self.sandbox.resolve(path))
        entry = self._files.get(rel)
        if entry is None:
            if rel in self._skipped:
                raise ValueError(f"{rel} was skipped during ingest ({self._skipped[rel]})")
            raise ValueError(f"{rel} is not a file in the repository")
        if entry.kind != want:
            hint = "; use view_image" if entry.kind == "image" else ""
            raise ValueError(f"{rel} is {entry.kind}, not {want}{hint}")
        return rel, self.sandbox.root / rel

    # ---- tools ----
    def _t_list_dir(self, call_id: str, path: str, depth: int = 1) -> ToolExecution:
        rel = self.sandbox.relative(self.sandbox.resolve(path))
        if rel in self._skipped:
            raise ValueError(f"{rel} was skipped during ingest ({self._skipped[rel]})")
        prefix = "" if rel == "." else rel + "/"
        dirs: set[str] = set()
        files = []
        for f in self.repomap.files:
            if not f.path.startswith(prefix):
                continue
            parts = f.path[len(prefix):].split("/")
            if len(parts) <= depth:
                files.append({"path": f.path, "kind": f.kind, "size": f.size})
            for i in range(1, min(len(parts), depth + 1)):
                dirs.add(prefix + "/".join(parts[:i]) + "/")
        if not files and not dirs and rel != ".":
            raise ValueError(f"{rel} is not a directory in the repository")
        skipped = [{"path": s.path, "reason": s.reason} for s in self.repomap.skipped
                   if s.path.startswith(prefix) and s.path[len(prefix):].count("/") < depth]
        entries = [{"path": d, "kind": "dir"} for d in sorted(dirs)] + files
        out = {"path": rel, "entries": entries, "skipped": skipped, "truncated": False}
        text = json.dumps(out)
        while len(text.encode()) > MAX_RESULT_BYTES and out["entries"]:
            out["entries"] = out["entries"][: len(out["entries"]) * 3 // 4]
            out["truncated"] = True
            text = json.dumps(out)
        return self._ok(call_id, text)

    def _t_read_file(self, call_id: str, path: str, start_line: int = 1, max_lines: int = 400) -> ToolExecution:
        rel, full = self._entry(path, "text")
        lines = read_lines(full)
        total = len(lines)
        if total and start_line > total:
            raise ValueError(f"start_line {start_line} is past the end ({total} lines)")
        out, size, i, cut = [], 0, start_line - 1, False
        while i < total and len(out) < min(max_lines, 400):
            line = f"{i + 1}| {lines[i]}"
            n = _json_size(line)
            if size + n > _LINES_BUDGET:
                if not out:
                    out.append(_fit_line(line, _LINES_BUDGET))
                    cut = True
                    i += 1
                break
            out.append(line)
            size += n
            i += 1
        truncated = i < total or cut
        payload = {"path": rel, "total_lines": total, "lines": out,
                   "truncated": truncated, "next_start_line": i + 1 if truncated else None}
        text = json.dumps(payload, ensure_ascii=False)
        while len(text.encode()) > MAX_RESULT_BYTES and payload["lines"]:
            payload["lines"].pop()
            payload["truncated"] = True
            payload["next_start_line"] = len(payload["lines"]) + start_line
            text = json.dumps(payload, ensure_ascii=False)
        return self._ok(call_id, text)

    def _t_grep(self, call_id: str, pattern: str, path_glob: str | None = None, max_results: int = 50) -> ToolExecution:
        rx = re.compile(pattern)
        matches, truncated = [], False
        for f in self.repomap.files:
            if f.kind != "text" or (path_glob and not fnmatch.fnmatch(f.path, path_glob)):
                continue
            for n, line in enumerate(read_lines(self.sandbox.root / f.path), start=1):
                if rx.search(line[:2000]):
                    if len(matches) >= max_results:
                        truncated = True
                        break
                    matches.append({"file": f.path, "line": n, "text": line[:200]})
            if truncated:
                break
        payload = {"matches": matches, "truncated": truncated}
        text = json.dumps(payload, ensure_ascii=False)
        while len(text.encode()) > MAX_RESULT_BYTES and payload["matches"]:
            payload["matches"].pop()
            payload["truncated"] = True
            text = json.dumps(payload, ensure_ascii=False)
        return self._ok(call_id, text)

    def _t_read_json(self, call_id: str, path: str, pointer: str = "") -> ToolExecution:
        rel, full = self._entry(path, "text")
        value = resolve_pointer(load_data_decimal(full), pointer)
        text = dumps_decimal(value)
        if len(text.encode()) > MAX_RESULT_BYTES:
            if isinstance(value, dict):
                raise ValueError(f"value is too large; use a deeper pointer. Keys here: {list(value)[:100]}")
            if isinstance(value, list):
                raise ValueError(f"value is a {len(value)}-item array; use a deeper pointer such as {pointer}/0")
            raise ValueError("value is too large")
        return self._ok(call_id, f'{{"path": {json.dumps(rel)}, "pointer": {json.dumps(pointer)}, "value": {text}}}')

    def _t_view_image(self, call_id: str, path: str) -> ToolExecution:
        rel, full = self._entry(path, "image")
        with Image.open(full) as im:
            w0, h0 = im.size
            im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
            im.thumbnail((IMAGE_LONG_EDGE, IMAGE_LONG_EDGE))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            w, h = im.size
        note = (f"{rel}: original {w0}x{h0}, shown {w}x{h}. Highlight rects are [x, y, w, h] fractions "
                f"(0-1) of the image.")
        return ToolExecution(ToolResultPart(call_id, [TextPart(note), ImagePart(buf.getvalue(), "image/png", rel)]))

    def _t_submit_storyboard(self, call_id: str, storyboard: Any) -> ToolExecution:
        if isinstance(storyboard, str):
            try:
                storyboard = json.loads(storyboard)
            except ValueError:
                pass
        outcome = self._submit(storyboard)
        return ToolExecution(ToolResultPart(call_id, [TextPart(outcome.feedback)], is_error=not outcome.accepted),
                             outcome.storyboard)
