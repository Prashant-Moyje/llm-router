"""Model adapters.

Two implementations behind one interface:

* ``AnthropicProvider`` - real API calls, real token usage, real wall-clock.
* ``MockProvider``      - deterministic, free, offline. Used by tests and by
  anyone who wants to exercise the full pipeline without spending money.

Cost is always computed from the token counts the API *reports*, never from a
local tokenizer estimate. Claude models do not share a tokenizer (Sonnet 5 uses
a different one from Haiku 4.5 and produces roughly 30% more tokens for the
same text), so a single local tokenizer would mis-bill at least one tier and
quietly distort the whole cost comparison this project exists to measure.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass

from .config import ModelSpec


@dataclass(frozen=True)
class Generation:
    """One model call: what it said, what it cost, how long it took."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    model_id: str
    error: str | None = None


class Provider:
    def generate(self, prompt: str, spec: ModelSpec, system: str = "") -> Generation:
        raise NotImplementedError


class AnthropicProvider(Provider):
    """Calls the Claude Messages API.

    Requires ANTHROPIC_API_KEY in the environment. Install with
    ``pip install anthropic``.
    """

    def __init__(self, api_key: str | None = None, max_retries: int = 3) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "The anthropic package is required. pip install anthropic"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._client = Anthropic(api_key=key, max_retries=max_retries)

    def generate(self, prompt: str, spec: ModelSpec, system: str = "") -> Generation:
        kwargs: dict = {
            "model": spec.model_id,
            "max_tokens": spec.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        # Newer models reject temperature/top_p entirely. Sending it anyway is a
        # 400, not a warning, so this has to be a per-model capability flag.
        if spec.supports_temperature:
            kwargs["temperature"] = spec.temperature

        start = time.perf_counter()
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - record, don't crash the sweep
            return Generation(
                text="",
                input_tokens=0,
                output_tokens=0,
                latency_s=time.perf_counter() - start,
                cost_usd=0.0,
                model_id=spec.model_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency = time.perf_counter() - start

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        return Generation(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            cost_usd=spec.cost_usd(in_tok, out_tok),
            model_id=spec.model_id,
        )


class MockProvider(Provider):
    """Deterministic fake used for pipeline tests and cost-free dry runs.

    It fabricates a capability gap: the "small" tier answers easy items
    correctly and fails a controlled share of hard ones. This exists so the
    plumbing can be validated end to end. Numbers it produces are not evidence
    of anything and must never be reported as results.
    """

    def __init__(self, small_hard_accuracy: float = 0.35, latency_scale: float = 1.0):
        self.small_hard_accuracy = small_hard_accuracy
        self.latency_scale = latency_scale

    @staticmethod
    def _unit_hash(text: str) -> float:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def generate(self, prompt: str, spec: ModelSpec, system: str = "") -> Generation:
        # The answer key is embedded in the prompt by the synthetic task builder.
        marker = "[[ANSWER="
        answer = ""
        difficulty = "easy"
        if marker in prompt:
            answer = prompt.split(marker, 1)[1].split("]]", 1)[0]
        if "[[HARD]]" in prompt:
            difficulty = "hard"

        is_small = spec.input_usd_per_mtok <= 1.0
        roll = self._unit_hash(prompt + spec.model_id)
        if not is_small:
            correct = True
        elif difficulty == "easy":
            correct = roll < 0.97
        else:
            correct = roll < self.small_hard_accuracy

        text = answer if correct else "0"
        in_tok = max(1, len(prompt) // 4)
        out_tok = max(1, len(text) // 4)
        latency = (0.35 if is_small else 1.6) * self.latency_scale
        return Generation(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            cost_usd=spec.cost_usd(in_tok, out_tok),
            model_id=spec.model_id,
        )


class OpenAICompatProvider(Provider):
    """Any OpenAI-compatible /chat/completions endpoint.

    Covers Groq, OpenRouter, Together, DeepInfra, local vLLM, and Ollama's
    compatibility endpoint. One adapter instead of one per vendor, because the
    router does not care where tokens come from - it needs two tiers with a
    real capability gap and reported token usage.

    Free tiers are aggressively rate-limited, so 429 handling is not optional
    here. A 429 is retried with backoff, honouring Retry-After when present; it
    is not recorded as a model failure, because a throttled request says nothing
    about model quality and dropping it would bias the labels toward whichever
    tier is throttled harder.
    """

    def __init__(
        self,
        base_url: str,
        api_key_env: str = "GROQ_API_KEY",
        max_retries: int = 10,
        timeout_s: float = 120.0,
        require_key: bool = True,
    ) -> None:
        import httpx

        key = os.environ.get(api_key_env, "")
        if require_key and not key:
            raise RuntimeError(
                f"{api_key_env} is not set. Get a free key at console.groq.com "
                f"(or set a different api_key_env in your config)."
            )
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {key or 'not-needed'}"},
        )

    def generate(self, prompt: str, spec: ModelSpec, system: str = "") -> Generation:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload: dict = {
            "model": spec.model_id,
            "messages": messages,
            "max_tokens": spec.max_tokens,
        }
        if spec.supports_temperature:
            payload["temperature"] = spec.temperature
        payload.update(spec.extra_params)

        wall_start = time.perf_counter()
        attempt_start = wall_start
        last_err = "unknown"
        for attempt in range(self.max_retries):
            # Restarted per attempt so that backoff sleep and failed attempts
            # are excluded from the reported latency. Without this, a single
            # 429 on a rate-limited free tier records multi-second latency for
            # a request the model answered in milliseconds, and the latency
            # comparison between tiers becomes a measurement of throttling.
            attempt_start = time.perf_counter()
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2**attempt, 30))
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("retry-after")
                # Jitter matters here: without it, concurrent workers throttled
                # at the same instant wake together and re-collide, so the
                # retry budget burns on synchronised retries rather than
                # spreading load. Retry-After is a floor, not a ceiling.
                base = float(retry_after) if retry_after else min(2**attempt, 60)
                wait = base + random.uniform(0, min(base, 5.0))
                last_err = f"HTTP {resp.status_code}"
                time.sleep(min(wait, 90))
                continue

            if resp.status_code != 200:
                return Generation(
                    "", 0, 0, time.perf_counter() - attempt_start, 0.0,
                    spec.model_id,
                    error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                )

            data = resp.json()
            latency = time.perf_counter() - attempt_start
            text = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage", {}) or {}
            in_tok = int(usage.get("prompt_tokens", 0))
            out_tok = int(usage.get("completion_tokens", 0))
            return Generation(
                text=text,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_s=latency,
                cost_usd=spec.cost_usd(in_tok, out_tok),
                model_id=spec.model_id,
            )

        return Generation(
            "", 0, 0, time.perf_counter() - wall_start, 0.0, spec.model_id,
            error=f"exhausted {self.max_retries} retries: {last_err}",
        )


def get_provider(name: str, cfg=None) -> Provider:
    if name == "anthropic":
        return AnthropicProvider()
    if name == "mock":
        return MockProvider()
    if name in ("groq", "openai_compat", "openrouter"):
        if cfg is None:
            raise ValueError(f"provider {name!r} needs a Config for base_url")
        return OpenAICompatProvider(cfg.base_url, cfg.api_key_env)
    if name == "ollama":
        base = cfg.base_url if cfg else "http://localhost:11434/v1"
        return OpenAICompatProvider(base, "OLLAMA_KEY", require_key=False)
    raise ValueError(f"Unknown provider: {name!r}")
