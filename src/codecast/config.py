"""Configuration: packaged defaults.yaml deep-merged with an optional user file."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from codecast.errors import ConfigError

KNOWN_PROVIDERS = ("anthropic", "openai")
DEFAULT_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


class LLMConfig(BaseModel):
    provider: str
    model: str | None
    api_key_env: str
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout_s: float = 120
    extended_thinking: bool = False
    caps: dict[str, Any] = {}


class AgentConfig(BaseModel):
    max_steps: int | None = None
    max_total_tokens: int = 600_000
    wall_clock_s: float = 240
    revision_steps: int = 15


class VoiceConfig(BaseModel):
    piper_voice: str = "en_US-lessac-medium"
    voices_dir: str = "~/.cache/codecast/voices"
    say_voice: str = "Samantha"
    edge_voice: str = "en-US-AriaNeural"
    gap_s: float = 0.15
    tail_s: float = 0.5

    @property
    def voices_path(self) -> Path:
        return Path(self.voices_dir).expanduser()


class RenderConfig(BaseModel):
    terminal_output_allowlist: list[str]


class ModelRate(BaseModel):
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float


class Config(BaseModel):
    llm: LLMConfig
    agent: AgentConfig
    voice: VoiceConfig
    render: RenderConfig
    pricing: dict[str, ModelRate] = {}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _defaults() -> dict:
    text = resources.files("codecast").joinpath("defaults.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def load_config(path: Path | None = None) -> Config:
    data = _defaults()
    user: dict = {}
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        try:
            user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML in {path}: {e}") from e
        if not isinstance(user, dict):
            raise ConfigError(f"config root must be a mapping: {path}")
        for section, default_value in data.items():
            if isinstance(default_value, dict) and section in user and not isinstance(user[section], dict):
                raise ConfigError(f"config section {section!r} must be a mapping in {path}")
        data = _deep_merge(data, user)

    user_llm = user.get("llm") or {}
    provider = data["llm"]["provider"]
    if provider not in KNOWN_PROVIDERS:
        raise ConfigError(f"unknown llm.provider {provider!r}; expected one of {KNOWN_PROVIDERS}")
    if provider != "anthropic" and not user_llm.get("model"):
        raise ConfigError(f"llm.model must be set explicitly when llm.provider is {provider!r}")
    if "api_key_env" not in user_llm:
        data["llm"]["api_key_env"] = DEFAULT_KEY_ENV[provider]
    if data["llm"].get("extended_thinking"):
        raise ConfigError("llm.extended_thinking is incompatible with forced tool_choice in Phase A")

    try:
        return Config.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"invalid configuration: {e}") from e
