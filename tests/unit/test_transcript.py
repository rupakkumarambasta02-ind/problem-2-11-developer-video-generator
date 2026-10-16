import json

from codecast.agent.transcript import Transcript, load_responses, message_from_json, message_to_json
from codecast.llm.fake import FakeProvider
from codecast.llm.types import (
    ChatRequest, ChatResponse, ImagePart, Message, TextPart, ToolCallPart, ToolResultPart, Usage,
)


def test_message_round_trip_with_image_sidecar(tmp_path):
    msg = Message("tool", [ToolResultPart("c1", [TextPart("note"), ImagePart(b"PNGDATA", "image/png", "a.png")])])
    d = message_to_json(msg, tmp_path / "images")
    ref = d["parts"][0]["parts"][1]
    assert ref["type"] == "image_ref" and ref["path"].startswith("images/") and ref["path"].endswith(".png")
    assert (tmp_path / ref["path"]).read_bytes() == b"PNGDATA"
    assert message_from_json(json.loads(json.dumps(d)), tmp_path) == msg


def test_transcript_logs_and_replays(tmp_path):
    path = tmp_path / "transcript.jsonl"
    t = Transcript(path)
    r1 = ChatResponse(Message("assistant", [ToolCallPart("c1", "list_dir", {"path": "."})]), "tool_use", Usage(5, 2, 1, 0))
    r2 = ChatResponse(Message("assistant", [TextPart("done")]), "end_turn", Usage(6, 3, 0, 0))
    req = ChatRequest(messages=[Message("user", [TextPart("hi")])])
    t.log_request(1, req, req.messages)
    t.log_response(1, r1)
    t.log_event(1, "duration_feedback", {"total": 63.0})
    t.log_response(2, r2)
    kinds = [json.loads(l)["type"] for l in path.read_text().split("\n") if l.strip()]
    assert kinds == ["request", "response", "event", "response"]
    assert load_responses(path) == [r1, r2]
    fp = FakeProvider.from_transcript(path)
    assert fp.complete(req) == r1 and fp.complete(req) == r2


def test_unicode_line_separators_in_content_round_trip(tmp_path):
    # U+0085 / U+2028 / U+2029 are not escaped by json.dumps(ensure_ascii=False) and must not split a record.
    odd = "a b c\x85d e f g h"
    path = tmp_path / "transcript.jsonl"
    t = Transcript(path)
    result = ToolResultPart("c1", [TextPart(odd)])
    req = ChatRequest(messages=[Message("tool", [result])])
    t.log_request(1, req, [Message("tool", [result])])
    resp = ChatResponse(Message("assistant", [TextPart(odd)]), "end_turn", Usage())
    t.log_response(1, resp)
    assert load_responses(path) == [resp]
    assert [json.loads(l)["type"] for l in path.read_text().split("\n") if l.strip()] == ["request", "response"]
