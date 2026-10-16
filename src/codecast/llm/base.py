"""LLMProvider protocol and the retry helper (3 attempts, exponential backoff with jitter)."""

from __future__ import annotations

import random
import time
from typing import Callable, Protocol

from codecast.llm.errors import RetryableError
from codecast.llm.types import ChatRequest, ChatResponse, ProviderCaps


class LLMProvider(Protocol):
    name: str
    model: str
    caps: ProviderCaps

    def complete(self, req: ChatRequest) -> ChatResponse: ...


def complete_with_retries(provider: LLMProvider, req: ChatRequest, *, attempts: int = 3,
                          base_delay: float = 1.0, sleep: Callable[[float], None] = time.sleep,
                          rng: Callable[[], float] = random.random) -> ChatResponse:
    for attempt in range(1, attempts + 1):
        try:
            return provider.complete(req)
        except RetryableError:
            if attempt == attempts:
                raise
            sleep(base_delay * 2 ** (attempt - 1) * (1 + rng()))
    raise AssertionError("unreachable")
