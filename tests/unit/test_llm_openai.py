import json

import httpx
import pytest
import respx

from codecast.llm.errors import AuthError, ContextLengthError, ProviderError, RetryableError
from codecast.llm.openai import OPENAI_CAPS, OpenAIProvider, build_request, parse_response
from codecast.llm.types import (
    ChatRequest, ImagePart, Message, ProviderCaps, TextPart, ToolCallPart, ToolResultPart, ToolSpec,
)

URL = "https://api.openai.com/v1/chat/completions"
TOOLS = [ToolSpec("view_image", "View", {"type": "object", "properties": {}})]


def conversation():
    return [
        Message("system", [TextPart("sys")]),
        Message("user", [TextPart("go")]),
        Message("assistant", [ToolCallPart("a", "view_image", {"path": "x.png"}),
                              ToolCallPart("b", "list_dir", {}),
                              ToolCallPart("c", "view_image", {"path": "y.png"})]),
        Message("tool", [
            ToolResultPart("a", [TextPart("x.png 10x10"), ImagePart(b"AAA", "image/png", "x.png")]),
            ToolResultPart("b", [TextPart("no such dir")], is_error=True),
            ToolResultPart("c", [ImagePart(b"BBB", "image/jpeg", "y.png")]),
        ]),
        Message("user", [TextPart("nudge")]),
    ]


def test_image_split_ordering():
    body = build_request(ChatRequest(messages=conversation(), tools=TOOLS), "m", OPENAI_CAPS, base_url=None)
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool", "tool", "user", "user"]
    tool_msgs = body["messages"][3:6]
    assert [t["tool_call_id"] for t in tool_msgs] == ["a", "b", "c"]
    assert tool_msgs[0]["content"] == "x.png 10x10\nimage x.png returned, see next message"
    assert tool_msgs[1]["content"] == "ERROR: no such dir"
    images = body["messages"][6]["content"]
    assert images[0] == {"type": "text", "text": "Result of a view_image path=x.png"}
    assert images[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert images[2] == {"type": "text", "text": "Result of c view_image path=y.png"}
    assert images[3]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert body["messages"][7] == {"role": "user", "content": "nudge"}
    call = body["messages"][2]["tool_calls"][0]
    assert call["function"] == {"name": "view_image", "arguments": json.dumps({"path": "x.png"})}


@pytest.mark.parametrize("choice,expected", [
    ("auto", "auto"), ("required", "required"),
    ({"name": "submit_storyboard"}, {"type": "function", "function": {"name": "submit_storyboard"}}),
])
def test_tool_choice(choice, expected):
    body = build_request(ChatRequest(messages=conversation()[:2], tools=TOOLS, tool_choice=choice),
                         "m", OPENAI_CAPS, base_url=None)
    assert body["tool_choice"] == expected


def test_token_param_and_temperature():
    req = ChatRequest(messages=conversation()[:2], max_tokens=50, temperature=0.3)
    assert build_request(req, "m", OPENAI_CAPS, base_url=None)["max_completion_tokens"] == 50
    compat = build_request(req, "m", OPENAI_CAPS, base_url="http://localhost:11434/v1")
    assert compat["max_tokens"] == 50 and "max_completion_tokens" not in compat
    no_temp = ProviderCaps(True, True, False, False, 128_000)
    assert "temperature" not in build_request(req, "m", no_temp, base_url=None)


def test_parse_response_invalid_json_args_and_usage():
    resp = parse_response({
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{bad json"}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 40}},
    })
    call = resp.message.tool_calls()[0]
    assert call.args == {"__invalid_json__": "{bad json"}
    assert resp.stop == "tool_use"
    assert (resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.cache_read_input_tokens) == (60, 7, 40)


def ok_body():
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}


@respx.mock
def test_end_to_end_with_sdk():
    respx.post(URL).mock(return_value=httpx.Response(200, json=ok_body()))
    resp = OpenAIProvider(api_key="k", model="m").complete(ChatRequest(messages=conversation()[:2]))
    assert resp.message.text() == "hi" and resp.stop == "end_turn"


def err(status, code, message):
    return httpx.Response(status, json={"error": {"message": message, "type": "invalid_request_error", "code": code}})


@pytest.mark.parametrize("response,exc", [
    (err(429, "rate_limit_exceeded", "slow"), RetryableError),
    (err(500, None, "boom"), RetryableError),
    (err(401, "invalid_api_key", "bad"), AuthError),
    (err(400, "context_length_exceeded", "This model's maximum context length is 128000 tokens"), ContextLengthError),
    (err(400, None, "bad request"), ProviderError),
])
@respx.mock
def test_errors_normalized(response, exc):
    respx.post(URL).mock(return_value=response)
    with pytest.raises(exc):
        OpenAIProvider(api_key="k", model="m").complete(ChatRequest(messages=conversation()[:2]))
