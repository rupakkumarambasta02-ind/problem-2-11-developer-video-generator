import pytest

from codecast.config import LLMConfig, ModelRate
from codecast.errors import ConfigError
from codecast.llm.base import complete_with_retries
from codecast.llm.errors import AuthError, ProviderError, RetryableError
from codecast.llm.fake import FakeProvider, say, tool_call
from codecast.llm.pricing import estimate_cost
from codecast.llm.registry import create_provider
from codecast.llm.types import ChatRequest, Message, TextPart, ToolCallPart, Usage


def req():
    return ChatRequest(messages=[Message("user", [TextPart("hi")])])


def test_usage_adds():
    a = Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40)
    assert a == Usage(11, 22, 33, 44)


def test_message_helpers():
    m = Message("assistant", [TextPart("a"), ToolCallPart("c1", "list_dir", {}), TextPart("b")])
    assert m.text() == "a\nb"
    assert [c.id for c in m.tool_calls()] == ["c1"]


def test_fake_provider_scripts_and_records():
    fp = FakeProvider([say("one"), tool_call("list_dir", {"path": "."}, "x")])
    assert fp.complete(req()).message.text() == "one"
    r2 = fp.complete(req())
    assert r2.stop == "tool_use" and r2.message.tool_calls()[0].args == {"path": "."}
    assert len(fp.requests) == 2
    with pytest.raises(ProviderError, match="exhausted"):
        fp.complete(req())


def test_fake_provider_raises_scripted_errors_and_callables():
    fp = FakeProvider([RetryableError("429"), lambda r: say(f"n={len(r.messages)}")])
    with pytest.raises(RetryableError):
        fp.complete(req())
    assert fp.complete(req()).message.text() == "n=1"


def test_retry_then_success():
    sleeps = []
    fp = FakeProvider([RetryableError("a"), RetryableError("b"), say("ok")])
    out = complete_with_retries(fp, req(), sleep=sleeps.append, rng=lambda: 0.5)
    assert out.message.text() == "ok"
    assert sleeps == [1.5, 3.0]


def test_retry_gives_up_after_three():
    fp = FakeProvider([RetryableError("a")] * 3)
    with pytest.raises(RetryableError):
        complete_with_retries(fp, req(), sleep=lambda s: None)


def test_auth_error_not_retried():
    fp = FakeProvider([AuthError("bad key"), say("never")])
    with pytest.raises(AuthError):
        complete_with_retries(fp, req(), sleep=lambda s: None)
    assert len(fp.requests) == 1


def test_estimate_cost():
    rates = {"m": ModelRate(input_per_mtok=3, output_per_mtok=15, cache_read_per_mtok=0.3, cache_write_per_mtok=3.75)}
    cost, why = estimate_cost("m", Usage(1_000_000, 100_000, 2_000_000, 0), rates)
    assert why is None and cost == pytest.approx(3 + 1.5 + 0.6)
    cost, why = estimate_cost("other", Usage(1, 1, 0, 0), rates)
    assert cost is None and "other" in why


def llm_cfg(**kw):
    base = dict(provider="anthropic", model="m", api_key_env="CODECAST_TEST_KEY")
    base.update(kw)
    return LLMConfig(**base)


def test_registry_missing_key_is_config_error():
    with pytest.raises(ConfigError, match="CODECAST_TEST_KEY"):
        create_provider(llm_cfg())


def test_registry_builds_adapters_without_network(monkeypatch):
    monkeypatch.setenv("CODECAST_TEST_KEY", "k")
    p = create_provider(llm_cfg())
    assert p.name == "anthropic" and p.model == "m"
    p = create_provider(llm_cfg(provider="openai", base_url="http://localhost:1/v1"))
    assert p.name == "openai"


def test_registry_require_key_false():
    p = create_provider(llm_cfg(), require_key=False)
    assert p.name == "anthropic"


def test_registry_unknown_provider():
    with pytest.raises(ConfigError):
        create_provider(llm_cfg(provider="nope"), require_key=False)
