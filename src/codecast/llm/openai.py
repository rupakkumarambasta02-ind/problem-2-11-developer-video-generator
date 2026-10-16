"""OpenAI Chat Completions adapter; also covers OpenAI-compatible endpoints via base_url (spec §4.2)."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from typing import Any

import openai as sdk

from codecast.llm.errors import AuthError, ContextLengthError, LLMError, ProviderError, RetryableError
from codecast.llm.types import (
    ChatRequest, ChatResponse, ImagePart, Message, ProviderCaps, TextPart, ToolCallPart, ToolResultPart, Usage,
)

OPENAI_CAPS = ProviderCaps(vision=True, tool_use=True, prompt_caching=False, supports_temperature=True,
                           context_window=128_000)


def _data_uri(p: ImagePart) -> str:
    return f"data:{p.media_type};base64,{base64.b64encode(p.data).decode()}"


def _user_content(parts) -> str | list[dict]:
    if all(isinstance(p, TextPart) for p in parts):
        return "\n".join(p.text for p in parts)
    out = []
    for p in parts:
        if isinstance(p, TextPart):
            out.append({"type": "text", "text": p.text})
        elif isinstance(p, ImagePart):
            out.append({"type": "image_url", "image_url": {"url": _data_uri(p)}})
    return out


def _tool_messages(m: Message) -> list[dict]:
    out, images = [], []
    for res in m.parts:
        if not isinstance(res, ToolResultPart):
            continue
        texts = []
        for q in res.parts:
            if isinstance(q, TextPart):
                texts.append(q.text)
            else:
                texts.append(f"image {q.source_path or 'image'} returned, see next message")
                images.append((res.call_id, q))
        content = "\n".join(texts)
        out.append({"role": "tool", "tool_call_id": res.call_id,
                    "content": f"ERROR: {content}" if res.is_error else content})
    if images:
        blocks = []
        for call_id, img in images:
            blocks.append({"type": "text", "text": f"Result of {call_id} view_image path={img.source_path}"})
            blocks.append({"type": "image_url", "image_url": {"url": _data_uri(img)}})
        out.append({"role": "user", "content": blocks})
    return out


def _choice(choice):
    if choice in ("auto", "required"):
        return choice
    return {"type": "function", "function": {"name": choice["name"]}}


def build_request(req: ChatRequest, model: str, caps: ProviderCaps, *, base_url: str | None) -> dict[str, Any]:
    msgs: list[dict] = []
    for m in req.messages:
        if m.role == "system":
            msgs.append({"role": "system", "content": m.text()})
        elif m.role == "user":
            msgs.append({"role": "user", "content": _user_content(m.parts)})
        elif m.role == "assistant":
            d: dict[str, Any] = {"role": "assistant", "content": m.text() or None}
            calls = m.tool_calls()
            if calls:
                d["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": json.dumps(c.args)}} for c in calls]
            msgs.append(d)
        else:
            msgs.extend(_tool_messages(m))
    body: dict[str, Any] = {"model": model, "messages": msgs}
    body["max_completion_tokens" if base_url is None else "max_tokens"] = req.max_tokens
    if req.tools:
        body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                           "parameters": t.input_schema}} for t in req.tools]
        body["tool_choice"] = _choice(req.tool_choice)
    if req.temperature is not None and caps.supports_temperature:
        body["temperature"] = req.temperature
    return body


def parse_response(data: dict) -> ChatResponse:
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    parts: list = []
    if msg.get("content"):
        parts.append(TextPart(msg["content"]))
    for tc in msg.get("tool_calls") or []:
        raw = tc["function"].get("arguments") or "{}"
        try:
            args = json.loads(raw)
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError:
            args = {"__invalid_json__": raw}
        parts.append(ToolCallPart(tc["id"], tc["function"]["name"], args))
    finish = choice.get("finish_reason")
    if any(isinstance(p, ToolCallPart) for p in parts):
        stop = "tool_use"
    elif finish == "length":
        stop = "max_tokens"
    else:
        stop = "end_turn"
    u = data.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    usage = Usage((u.get("prompt_tokens") or 0) - cached, u.get("completion_tokens") or 0, cached, 0)
    return ChatResponse(Message("assistant", parts), stop, usage)


def map_error(e: Exception) -> LLMError:
    if isinstance(e, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return AuthError(str(e))
    if isinstance(e, (sdk.RateLimitError, sdk.APIConnectionError, sdk.InternalServerError)):
        return RetryableError(str(e))
    if isinstance(e, sdk.APIStatusError):
        if e.status_code >= 500:
            return RetryableError(str(e))
        if getattr(e, "code", None) == "context_length_exceeded" or "maximum context length" in str(e):
            return ContextLengthError(str(e))
        return ProviderError(str(e))
    return ProviderError(f"{type(e).__name__}: {e}")


class OpenAIProvider:
    name = "openai"

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None,
                 caps_override: dict | None = None, client: Any = None):
        self.model = model
        self.base_url = base_url
        self.caps = replace(OPENAI_CAPS, **(caps_override or {}))
        self._client = client or sdk.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def complete(self, req: ChatRequest) -> ChatResponse:
        body = build_request(req, self.model, self.caps, base_url=self.base_url)
        try:
            completion = self._client.chat.completions.create(**body, timeout=req.timeout_s)
        except Exception as e:  # noqa: BLE001 - every SDK error is normalized
            raise map_error(e) from e
        return parse_response(completion.model_dump())
