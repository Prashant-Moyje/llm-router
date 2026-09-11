"""Serving layer.

Two endpoints, deliberately separated:

* ``POST /v1/route``    - decision only, no model call. ~3 ms p50 on one core.
* ``POST /v1/complete`` - decide, then execute against the chosen tier.

They are split because most teams adopting a router already have their own
inference path and want the decision, not the proxy. It also makes the router's
own latency measurable in isolation, which is the number that decides whether
the router is worth having at all.

Run:
    uvicorn llmrouter.serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .config import Config
from .providers import Provider, get_provider
from .router import LearnedRouter

CONFIG_PATH = os.environ.get("ROUTER_CONFIG", "configs/default.yaml")
MODEL_PATH = os.environ.get("ROUTER_MODEL", "artifacts/router.joblib")
DEFAULT_TAU = float(os.environ.get("ROUTER_TAU", "0.5"))
PROVIDER_NAME = os.environ.get("ROUTER_PROVIDER", "mock")

app = FastAPI(title="LLM cost/latency router", version="0.1.0")

_state: dict = {}

_METRICS = {
    "requests_total": 0,
    "routed_small_total": 0,
    "routed_large_total": 0,
    "cost_usd_total": 0.0,
    "router_latency_s_total": 0.0,
    "model_latency_s_total": 0.0,
}


class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1)
    tau: float | None = Field(default=None, ge=0.0, le=1.0)


class RouteResponse(BaseModel):
    tier: Literal["small", "large"]
    model_id: str
    p_small_sufficient: float
    tau: float
    router_latency_ms: float


class CompleteResponse(RouteResponse):
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model_latency_ms: float


@app.on_event("startup")
def _startup() -> None:
    cfg = Config.load(CONFIG_PATH)
    _state["cfg"] = cfg
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise RuntimeError(
            f"Router artifact missing at {model_file}. Run scripts/03_train_router.py."
        )
    _state["router"] = LearnedRouter.load(model_file)
    _state["provider"] = get_provider(PROVIDER_NAME, cfg)


def _decide(prompt: str, tau: float | None) -> tuple[RouteResponse, object]:
    cfg: Config = _state["cfg"]
    router: LearnedRouter = _state["router"]
    tau_v = DEFAULT_TAU if tau is None else tau

    t0 = time.perf_counter()
    p = float(router.predict_proba_small_ok([prompt])[0])
    router_ms = (time.perf_counter() - t0) * 1000

    use_small = p >= tau_v
    spec = cfg.small if use_small else cfg.large
    _METRICS["requests_total"] += 1
    _METRICS["routed_small_total" if use_small else "routed_large_total"] += 1
    _METRICS["router_latency_s_total"] += router_ms / 1000

    return (
        RouteResponse(
            tier="small" if use_small else "large",
            model_id=spec.model_id,
            p_small_sufficient=round(p, 4),
            tau=tau_v,
            router_latency_ms=round(router_ms, 3),
        ),
        spec,
    )


@app.post("/v1/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    decision, _ = _decide(req.prompt, req.tau)
    return decision


@app.post("/v1/complete", response_model=CompleteResponse)
def complete(req: RouteRequest) -> CompleteResponse:
    decision, spec = _decide(req.prompt, req.tau)
    provider: Provider = _state["provider"]
    gen = provider.generate(req.prompt, spec)
    if gen.error:
        raise HTTPException(status_code=502, detail=gen.error)

    _METRICS["cost_usd_total"] += gen.cost_usd
    _METRICS["model_latency_s_total"] += gen.latency_s
    return CompleteResponse(
        **decision.model_dump(),
        text=gen.text,
        input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens,
        cost_usd=round(gen.cost_usd, 8),
        model_latency_ms=round(gen.latency_s * 1000, 2),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": PROVIDER_NAME, "tau": DEFAULT_TAU}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus text exposition.

    Escalation rate and realised cost are the two things that drift in
    production: traffic mix shifts, the small model's hit rate falls, and the
    saving quietly evaporates. Those need to be on a dashboard, not in a README.
    """
    lines = [
        "# HELP llmrouter_requests_total Requests routed.",
        "# TYPE llmrouter_requests_total counter",
        f"llmrouter_requests_total {_METRICS['requests_total']}",
        "# TYPE llmrouter_routed_total counter",
        f'llmrouter_routed_total{{tier="small"}} {_METRICS["routed_small_total"]}',
        f'llmrouter_routed_total{{tier="large"}} {_METRICS["routed_large_total"]}',
        "# TYPE llmrouter_cost_usd_total counter",
        f"llmrouter_cost_usd_total {_METRICS['cost_usd_total']:.8f}",
        "# TYPE llmrouter_router_latency_seconds_total counter",
        f"llmrouter_router_latency_seconds_total {_METRICS['router_latency_s_total']:.6f}",
        "# TYPE llmrouter_model_latency_seconds_total counter",
        f"llmrouter_model_latency_seconds_total {_METRICS['model_latency_s_total']:.6f}",
    ]
    return "\n".join(lines) + "\n"
