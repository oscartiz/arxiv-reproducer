"""Per-run token and cost accounting.

Prices are USD per million tokens. Cache reads bill at ~0.1x the input rate,
cache writes (5-minute TTL) at 1.25x — the paper text is cached on every run,
so these terms dominate real-run costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-5": ModelPrice(3.00, 15.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class Usage:
    """Token totals accumulated across every message of a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    requests: int = field(default=0)

    def add(self, message_usage: object) -> None:
        """Accumulate one API message's usage object (fields may be None)."""
        if message_usage is None:
            return
        self.requests += 1
        for attr in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = getattr(message_usage, attr, None)
            if value:
                setattr(self, attr, getattr(self, attr) + value)

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "api_requests": self.requests,
        }


def estimate_cost_usd(usage: Usage, model: str) -> float | None:
    """Dollar estimate for a run; None when the model's price is unknown."""
    price = MODEL_PRICES.get(model)
    if price is None:
        return None
    dollars = (
        usage.input_tokens * price.input_per_mtok
        + usage.cache_read_input_tokens * price.input_per_mtok * CACHE_READ_MULTIPLIER
        + usage.cache_creation_input_tokens * price.input_per_mtok * CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000
    return round(dollars, 4)
