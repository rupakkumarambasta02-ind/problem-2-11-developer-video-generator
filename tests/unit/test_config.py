from pathlib import Path

import pytest

from codecast.config import load_config
from codecast.errors import ConfigError


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


def test_defaults_load():
    cfg = load_config()
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == "claude-sonnet-5-5"
    assert cfg.llm.temperature == 0.3
    assert cfg.llm.max_tokens == 4096
    assert cfg.llm.timeout_s == 120
    assert cfg.voice.gap_s == 0.15
    assert cfg.voice.tail_s == 0.5
    assert "*.log" in cfg.render.terminal_output_allowlist


def test_user_file_deep_merges(tmp_path):
    cfg = load_config(write(tmp_path, "llm:\n  temperature: 0.1\n"))
    assert cfg.llm.temperature == 0.1
    assert cfg.llm.model == "claude-sonnet-5-5"


def test_openai_requires_explicit_model(tmp_path):
    with pytest.raises(ConfigError, match="model"):
        load_config(write(tmp_path, "llm:\n  provider: openai\n"))


def test_openai_defaults_key_env(tmp_path):
    cfg = load_config(write(tmp_path, "llm:\n  provider: openai\n  model: some-model\n"))
    assert cfg.llm.api_key_env == "OPENAI_API_KEY"


def test_unknown_provider_rejected(tmp_path):
    with pytest.raises(ConfigError, match="provider"):
        load_config(write(tmp_path, "llm:\n  provider: nope\n  model: x\n"))


def test_extended_thinking_rejected(tmp_path):
    with pytest.raises(ConfigError, match="tool_choice"):
        load_config(write(tmp_path, "llm:\n  extended_thinking: true\n"))


def test_missing_file_is_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "absent.yaml")


def test_pricing_parsed(tmp_path):
    cfg = load_config(write(tmp_path, (
        "pricing:\n  m1:\n    input_per_mtok: 3\n    output_per_mtok: 15\n"
        "    cache_read_per_mtok: 0.3\n    cache_write_per_mtok: 3.75\n")))
    assert cfg.pricing["m1"].output_per_mtok == 15


def test_non_mapping_llm_section_is_config_error(tmp_path):
    with pytest.raises(ConfigError, match="llm"):
        load_config(write(tmp_path, "llm: foo\n"))


def test_null_llm_section_is_config_error(tmp_path):
    with pytest.raises(ConfigError, match="llm"):
        load_config(write(tmp_path, "llm: null\n"))


def test_list_voice_section_is_config_error(tmp_path):
    with pytest.raises(ConfigError, match="voice"):
        load_config(write(tmp_path, "voice: [1]\n"))
