"""Offline provider: replays scripted responses or a recorded transcript."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

from codecast.llm.errors import ProviderError
from codecast.llm.types import ChatRequest, ChatResponse, Message, ProviderCaps, TextPart, ToolCallPart, Usage

FAKE_CAPS = ProviderCaps(vision=True, tool_use=True, prompt_caching=False, supports_temperature=True,
                         context_window=200_000)
ScriptItem = Union[ChatResponse, Callable[[ChatRequest], ChatResponse], Exception]


class FakeProvider:
    name = "fake"

    def __init__(self, script: list[ScriptItem], *, caps: ProviderCaps = FAKE_CAPS, model: str = "fake-model"):
        self._script = list(script)
        self.caps = caps
        self.model = model
        self.requests: list[ChatRequest] = []

    def complete(self, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        if not self._script:
            raise ProviderError("FakeProvider script exhausted")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(req) if callable(item) else item

    @classmethod
    def from_transcript(cls, path: Path, **kw) -> "FakeProvider":
        from codecast.agent.transcript import load_responses
        return cls(load_responses(Path(path)), **kw)


def tool_call(name: str, args: dict, call_id: str = "c1", usage: Usage | None = None) -> ChatResponse:
    return ChatResponse(Message("assistant", [ToolCallPart(call_id, name, args)]), "tool_use", usage or Usage())


def say(text: str, usage: Usage | None = None) -> ChatResponse:
    return ChatResponse(Message("assistant", [TextPart(text)]), "end_turn", usage or Usage())
