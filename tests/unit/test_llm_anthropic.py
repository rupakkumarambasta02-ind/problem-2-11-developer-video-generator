import httpx
import pytest
import respx

from codecast.llm.anthropic import ANTHROPIC_CAPS, AnthropicProvider, build_request, parse_response
from codecast.llm.errors import AuthError, ContextLengthError, ProviderError, RetryableError
from codecast.llm.types import (
    ChatRequest, ImagePart, Message, TextPart, ToolCallPart, ToolResultPart, ToolSpec,
)

URL = "https://api.anthropic.com/v1/messages"

TOOLS = [ToolSpec("list_dir", "List", {"type": "object", "properties": {}}),
         ToolSpec("submit_storyboard", "Submit", {"type": "object", "properties": {}})]


def conversation():
    return [
        Message("system", [TextPart("You are the director.")]),
        Message("user", [TextPart("Explain this repo.")]),
        Message("assistant", [TextPart("Looking."), ToolCallPart("t1", "list_dir", {"path": "."}),
                              ToolCallPart("t2", "view_image", {"path": "a.png"})]),
        Message("tool", [ToolResultPart("t1", [TextPart("[]")]),
                         ToolResultPart("t2", [ImagePart(b"\x89PNG", "image/png", "a.png")], is_error=False)]),
        Message("user", [TextPart("Finish by calling submit_storyboard.")]),
    ]


def test_build_request_maps_roles_and_caching():
    body = build_request(ChatRequest(messages=conversation(), tools=TOOLS, tool_choice={"name": "submit_storyboard"},
                                     max_tokens=100, temperature=0.3), "m", ANTHROPIC_CAPS)
    assert body["system"] == [{"type": "text", "text": "You are the director.", "cache_control": {"type": "ephemeral"}}]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assistant = body["messages"][1]["content"]
    assert assistant[1] == {"type": "tool_use", "id": "t1", "name": "list_dir", "input": {"path": "."}}
    last = body["messages"][2]["content"]
    assert [b["type"] for b in last] == ["tool_result", "tool_result", "text"]
    assert last[1]["content"][0]["type"] == "image"
    assert last[1]["content"][0]["source"]["media_type"] == "image/png"
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in body["tools"][0]
    assert body["tool_choice"] == {"type": "tool", "name": "submit_storyboard"}
    assert body["temperature"] == 0.3 and body["max_tokens"] == 100


@pytest.mark.parametrize("choice,expected", [("auto", {"type": "auto"}), ("required", {"type": "any"})])
def test_tool_choice(choice, expected):
    body = build_request(ChatRequest(messages=conversation(), tools=TOOLS, tool_choice=choice), "m", ANTHROPIC_CAPS)
    assert body["tool_choice"] == expected


def test_parse_response_with_cache_usage():
    resp = parse_response({
        "content": [{"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "t9", "name": "grep", "input": {"pattern": "x"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3,
                  "cache_creation_input_tokens": 2},
    })
    assert resp.stop == "tool_use"
    assert resp.message.tool_calls()[0].args == {"pattern": "x"}
    assert (resp.usage.input_tokens, resp.usage.cache_read_input_tokens, resp.usage.cache_write_input_tokens) == (10, 3, 2)


def ok_body():
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 4, "output_tokens": 1}}


@respx.mock
def test_end_to_end_with_sdk():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=ok_body()))
    p = AnthropicProvider(api_key="k", model="m")
    resp = p.complete(ChatRequest(messages=conversation()[:2]))
    assert resp.message.text() == "hello" and resp.stop == "end_turn"
    assert route.called


def err(status, etype, message):
    return httpx.Response(status, json={"type": "error", "error": {"type": etype, "message": message}})


@pytest.mark.parametrize("response,exc", [
    (err(429, "rate_limit_error", "slow down"), RetryableError),
    (err(500, "api_error", "boom"), RetryableError),
    (err(529, "overloaded_error", "busy"), RetryableError),
    (err(401, "authentication_error", "bad key"), AuthError),
    (err(400, "invalid_request_error", "prompt is too long: 250000 tokens > 200000 maximum"), ContextLengthError),
    (err(400, "invalid_request_error", "bad schema"), ProviderError),
])
@respx.mock
def test_errors_normalized(response, exc):
    respx.post(URL).mock(return_value=response)
    with pytest.raises(exc):
        AnthropicProvider(api_key="k", model="m").complete(ChatRequest(messages=conversation()[:2]))


@respx.mock
def test_timeout_is_retryable():
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("t"))
    with pytest.raises(RetryableError):
        AnthropicProvider(api_key="k", model="m").complete(ChatRequest(messages=conversation()[:2]))
