"""Cost from configured per-model rates. Unknown model → cost None with a reason (never guessed)."""

from __future__ import annotations

from codecast.config import ModelRate
from codecast.llm.types import Usage


def estimate_cost(model: str, usage: Usage, rates: dict[str, ModelRate]) -> tuple[float | None, str | None]:
    rate = rates.get(model)
    if rate is None:
        return None, f"no pricing entry for model {model!r} in config 'pricing'"
    cost = (usage.input_tokens * rate.input_per_mtok + usage.output_tokens * rate.output_per_mtok
            + usage.cache_read_input_tokens * rate.cache_read_per_mtok
            + usage.cache_write_input_tokens * rate.cache_write_per_mtok) / 1_000_000
    return round(cost, 6), None
