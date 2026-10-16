"""JSONL transcript of every agent request/response; images stored once as sidecar files (spec §5.1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from codecast.llm.types import (
    ChatRequest, ChatResponse, ImagePart, Message, TextPart, ToolCallPart, ToolResultPart, Usage,
)

_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}


def _part_to_json(p, images_dir: Path) -> dict:
    if isinstance(p, TextPart):
        return {"type": "text", "text": p.text}
    if isinstance(p, ImagePart):
        name = f"{hashlib.sha256(p.data).hexdigest()}.{_EXT.get(p.media_type, 'bin')}"
        images_dir.mkdir(parents=True, exist_ok=True)
        target = images_dir / name
        if not target.exists():
            target.write_bytes(p.data)
        return {"type": "image_ref", "path": f"images/{name}", "media_type": p.media_type,
                "source_path": p.source_path}
    if isinstance(p, ToolCallPart):
        return {"type": "tool_call", "id": p.id, "name": p.name, "args": p.args}
    if isinstance(p, ToolResultPart):
        return {"type": "tool_result", "call_id": p.call_id, "is_error": p.is_error,
                "parts": [_part_to_json(q, images_dir) for q in p.parts]}
    raise TypeError(type(p).__name__)


def _part_from_json(d: dict, base_dir: Path):
    t = d["type"]
    if t == "text":
        return TextPart(d["text"])
    if t == "image_ref":
        return ImagePart((base_dir / d["path"]).read_bytes(), d["media_type"], d.get("source_path"))
    if t == "tool_call":
        return ToolCallPart(d["id"], d["name"], d["args"])
    if t == "tool_result":
        return ToolResultPart(d["call_id"], [_part_from_json(q, base_dir) for q in d["parts"]], d["is_error"])
    raise ValueError(f"unknown part type {t!r}")


def message_to_json(msg: Message, images_dir: Path) -> dict:
    return {"role": msg.role, "parts": [_part_to_json(p, images_dir) for p in msg.parts]}


def message_from_json(d: dict, base_dir: Path) -> Message:
    return Message(d["role"], [_part_from_json(p, base_dir) for p in d["parts"]])


class Transcript:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.path.parent / "images"

    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_request(self, step: int, req: ChatRequest, new_messages: list[Message]) -> None:
        self._write({"type": "request", "step": step, "tool_choice": req.tool_choice,
                     "n_messages": len(req.messages), "tools": [t.name for t in req.tools],
                     "messages": [message_to_json(m, self.images_dir) for m in new_messages]})

    def log_response(self, step: int, resp: ChatResponse) -> None:
        self._write({"type": "response", "step": step, "stop": resp.stop, "usage": asdict(resp.usage),
                     "message": message_to_json(resp.message, self.images_dir)})

    def log_messages(self, step: int, messages: list[Message]) -> None:
        self._write({"type": "messages", "step": step,
                     "messages": [message_to_json(m, self.images_dir) for m in messages]})

    def log_event(self, step: int, kind: str, data: dict) -> None:
        self._write({"type": "event", "step": step, "kind": kind, "data": data})


def load_responses(path: Path) -> list[ChatResponse]:
    path = Path(path)
    out = []
    # Split on "\n" only: str.splitlines() also breaks on U+0085, U+2028 and U+2029, which json.dumps
    # leaves raw under ensure_ascii=False, and tool results carry untrusted repo text.
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["type"] == "response":
            out.append(ChatResponse(message_from_json(rec["message"], path.parent), rec["stop"],
                                    Usage(**rec["usage"])))
    return out
