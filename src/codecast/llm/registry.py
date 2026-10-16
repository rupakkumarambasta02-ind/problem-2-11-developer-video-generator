"""Provider name → adapter factory. A new provider = one adapter module + one entry here."""

from __future__ import annotations

import os
from typing import Callable

from codecast.config import LLMConfig
from codecast.errors import ConfigError
from codecast.llm.base import LLMProvider


def _anthropic(cfg: LLMConfig, api_key: str) -> LLMProvider:
    from codecast.llm.anthropic import AnthropicProvider
    return AnthropicProvider(api_key=api_key, model=cfg.model, base_url=cfg.base_url, caps_override=cfg.caps)


def _openai(cfg: LLMConfig, api_key: str) -> LLMProvider:
    from codecast.llm.openai import OpenAIProvider
    return OpenAIProvider(api_key=api_key, model=cfg.model, base_url=cfg.base_url, caps_override=cfg.caps)


PROVIDERS: dict[str, Callable[[LLMConfig, str], LLMProvider]] = {"anthropic": _anthropic, "openai": _openai}


def create_provider(cfg: LLMConfig, *, require_key: bool = True) -> LLMProvider:
    factory = PROVIDERS.get(cfg.provider)
    if factory is None:
        raise ConfigError(f"unknown llm.provider {cfg.provider!r}")
    api_key = os.environ.get(cfg.api_key_env, "")
    if require_key and not api_key:
        raise ConfigError(f"{cfg.api_key_env} is not set; export it or set llm.api_key_env in your config")
    return factory(cfg, api_key or "missing-key")
