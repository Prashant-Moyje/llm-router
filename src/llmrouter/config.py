"""Configuration objects.

Prices are config, never hardcoded in logic. Provider pricing changes; a
project that bakes $/MTok into source produces stale numbers silently.
Verify rates against https://platform.claude.com/docs/en/about-claude/pricing
before quoting any dollar figure in a README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    """One model tier and its billing rates."""

    name: str
    model_id: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    max_tokens: int = 1024
    # Some models (e.g. Claude Sonnet 5 / Opus 5) reject non-default sampling
    # params. Keep this per-model rather than assuming a single API shape.
    supports_temperature: bool = True
    temperature: float = 0.0
    # "list_price"  - billed at the provider's published rate (what you'd pay)
    # "imputed"     - inference was free (free tier / local), dollars are the
    #                 published rate applied to measured token counts
    # "local"       - self-hosted; per-token price is a modelling assumption
    # Carried into reports so a dollar figure can never be mistaken for a bill.
    pricing_basis: str = "list_price"
    # Provider-specific request parameters merged into the payload, e.g.
    # {"reasoning_effort": "low"} for gpt-oss on Groq.
    #
    # This is a real routing lever, not a tuning knob. A reasoning model's
    # output length is the dominant cost term when input:output pricing is
    # ~1:5, so effort level moves cost more than tier choice does. Leaving both
    # tiers at default effort is how a "cheap" tier ends up costing MORE than
    # the expensive one - the weaker model compensates with longer chains.
    extra_params: dict = field(default_factory=dict)

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_mtok
            + output_tokens * self.output_usd_per_mtok
        ) / 1_000_000


@dataclass(frozen=True)
class Paths:
    data: Path = Path("data")
    artifacts: Path = Path("artifacts")
    reports: Path = Path("reports")

    def ensure(self) -> None:
        for p in (self.data, self.artifacts, self.reports):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    small: ModelSpec
    large: ModelSpec
    paths: Paths = field(default_factory=Paths)
    seed: int = 13
    # Fraction of the labelled set held out for threshold selection + reporting.
    test_size: float = 0.3
    # An item counts as "small model sufficed" when its score is within this
    # margin of the large model's score. 0.0 == small must match or beat large.
    sufficiency_margin: float = 0.0
    # For OpenAI-compatible endpoints (Groq, OpenRouter, Together, vLLM, Ollama).
    base_url: str = "https://api.groq.com/openai/v1"
    api_key_env: str = "GROQ_API_KEY"

    @staticmethod
    def load(path: str | Path) -> "Config":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        paths_raw = raw.get("paths", {})
        provider = raw.get("provider", {})
        return Config(
            small=ModelSpec(**raw["small"]),
            large=ModelSpec(**raw["large"]),
            paths=Paths(**{k: Path(v) for k, v in paths_raw.items()}),
            seed=int(raw.get("seed", 13)),
            test_size=float(raw.get("test_size", 0.3)),
            sufficiency_margin=float(raw.get("sufficiency_margin", 0.0)),
            base_url=provider.get("base_url", "https://api.groq.com/openai/v1"),
            api_key_env=provider.get("api_key_env", "GROQ_API_KEY"),
        )
