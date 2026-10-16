"""Anthropic Messages API adapter (spec §4.2)."""

from __future__ import annotations

import base64
from dataclasses import replace
from typing import Any

import anthropic as sdk

from codecast.llm.errors import AuthError, ContextLengthError, LLMError, ProviderError, RetryableError
from codecast.llm.types import (
    ChatRequest, ChatResponse, ImagePart, Message, ProviderCaps, TextPart, ToolCallPart, ToolResultPart, Usage,
)

ANTHROPIC_CAPS = ProviderCaps(vision=True, tool_use=True, prompt_caching=True, supports_temperature=True,
                              context_window=200_000)
_CACHE = {"type": "ephemeral"}


def _inner(p: TextPart | ImagePart) -> dict:
    if isinstance(p, ImagePart):
        return {"type": "image", "source": {"type": "base64", "media_type": p.media_type,
                                            "data": base64.b64encode(p.data).decode()}}
    return {"type": "text", "text": p.text}


def _block(p) -> dict | None:
    if isinstance(p, TextPart):
        return {"type": "text", "text": p.text} if p.text else None
    if isinstance(p, ImagePart):
        return _inner(p)
    if isinstance(p, ToolCallPart):
        return {"type": "tool_use", "id": p.id, "name": p.name, "input": p.args}
    if isinstance(p, ToolResultPart):
        return {"type": "tool_result", "tool_use_id": p.call_id,
                "content": [_inner(q) for q in p.parts] or [{"type": "text", "text": "(empty)"}],
                "is_error": p.is_error}
    raise TypeError(f"unsupported part {type(p).__name__}")


def _choice(choice) -> dict:
    if choice == "auto":
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    return {"type": "tool", "name": choice["name"]}


def build_request(req: ChatRequest, model: str, caps: ProviderCaps) -> dict[str, Any]:
    system = "\n\n".join(m.text() for m in req.messages if m.role == "system")
    msgs: list[dict] = []
    for m in req.messages:
        if m.role == "system":
            continue
        role = "assistant" if m.role == "assistant" else "user"
        blocks = [b for b in (_block(p) for p in m.parts) if b is not None]
        if not blocks:
            continue
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"].extend(blocks)
        else:
            msgs.append({"role": role, "content": blocks})
    body: dict[str, Any] = {"model": model, "max_tokens": req.max_tokens, "messages": msgs}
    if system:
        block = {"type": "text", "text": system}
        if caps.prompt_caching:
            block["cache_control"] = _CACHE
        body["system"] = [block]
    if req.tools:
        tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in req.tools]
        if caps.prompt_caching:
            tools[-1]["cache_control"] = _CACHE
        body["tools"] = tools
        body["tool_choice"] = _choice(req.tool_choice)
    if req.temperature is not None and caps.supports_temperature:
        body["temperature"] = req.temperature
    return body


def parse_response(data: dict) -> ChatResponse:
    parts = []
    for b in data.get("content") or []:
        if b.get("type") == "text" and b.get("text"):
            parts.append(TextPart(b["text"]))
        elif b.get("type") == "tool_use":
            parts.append(ToolCallPart(b["id"], b["name"], b.get("input") or {}))
    stop = {"tool_use": "tool_use", "max_tokens": "max_tokens"}.get(data.get("stop_reason"), "end_turn")
    u = data.get("usage") or {}
    usage = Usage(u.get("input_tokens") or 0, u.get("output_tokens") or 0,
                  u.get("cache_read_input_tokens") or 0, u.get("cache_creation_input_tokens") or 0)
    return ChatResponse(Message("assistant", parts), stop, usage)


def map_error(e: Exception) -> LLMError:
    if isinstance(e, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return AuthError(str(e))
    if isinstance(e, (sdk.RateLimitError, sdk.APIConnectionError, sdk.InternalServerError)):
        return RetryableError(str(e))
    if isinstance(e, sdk.APIStatusError):
        if e.status_code >= 500:
            return RetryableError(str(e))
        text = str(e).lower()
        if e.status_code in (400, 413) and ("prompt is too long" in text or "context window" in text
                                            or "too many tokens" in text):
            return ContextLengthError(str(e))
        return ProviderError(str(e))
    return ProviderError(f"{type(e).__name__}: {e}")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None,
                 caps_override: dict | None = None, client: Any = None):
        self.model = model
        self.caps = replace(ANTHROPIC_CAPS, **(caps_override or {}))
        self._client = client or sdk.Anthropic(api_key=api_key, base_url=base_url, max_retries=0)

    def complete(self, req: ChatRequest) -> ChatResponse:
        body = build_request(req, self.model, self.caps)
        try:
            # This SDK's create() has no `temperature` kwarg; send it through extra_body.
            extra = {"temperature": body.pop("temperature")} if "temperature" in body else None
            msg = self._client.messages.create(**body, extra_body=extra, timeout=req.timeout_s)
        except Exception as e:  # noqa: BLE001 - every SDK error is normalized
            raise map_error(e) from e
        return parse_response(msg.model_dump())
