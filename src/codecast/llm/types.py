"""Provider-neutral chat types (spec §4.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    data: bytes
    media_type: str
    source_path: str | None = None


@dataclass
class ToolCallPart:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResultPart:
    call_id: str
    parts: list[Union[TextPart, ImagePart]]
    is_error: bool = False


ContentPart = Union[TextPart, ImagePart, ToolCallPart, ToolResultPart]
Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    parts: list[ContentPart]

    def text(self) -> str:
        return "\n".join(p.text for p in self.parts if isinstance(p, TextPart))

    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.parts if isinstance(p, ToolCallPart)]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


ToolChoice = Union[Literal["auto", "required"], dict]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens,
                     self.cache_read_input_tokens + other.cache_read_input_tokens,
                     self.cache_write_input_tokens + other.cache_write_input_tokens)

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_write_input_tokens


@dataclass
class ChatRequest:
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    tool_choice: ToolChoice = "auto"
    max_tokens: int = 4096
    temperature: float | None = 0.3
    timeout_s: float = 120


@dataclass
class ChatResponse:
    message: Message
    stop: Literal["end_turn", "tool_use", "max_tokens"]
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class ProviderCaps:
    vision: bool
    tool_use: bool
    prompt_caching: bool
    supports_temperature: bool
    context_window: int
