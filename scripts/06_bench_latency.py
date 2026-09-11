"""Benchmark the router's own inference latency.

    python scripts/06_bench_latency.py

The router's latency is an overhead it charges on every single request,
including the ones it sends to the expensive tier. Any latency saving claimed
in the README has to be net of this number, so it gets measured rather than
assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmrouter.config import Config  # noqa: E402
from llmrouter.router import HeuristicRouter, LearnedRouter  # noqa: E402

PROMPT = (
    "Summarise the following support ticket and decide whether it needs "
    "escalation to engineering. Explain your reasoning briefly."
)


def bench(fn, n: int = 300) -> dict:
    fn()  # warm up: first call pays import and cache costs
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {
        "p50_ms": round(float(np.percentile(ts, 50)), 3),
        "p95_ms": round(float(np.percentile(ts, 95)), 3),
        "p99_ms": round(float(np.percentile(ts, 99)), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    learned = LearnedRouter.load(cfg.paths.artifacts / "router.joblib")
    heur = HeuristicRouter()

    out = {
        "learned_single": bench(lambda: learned.predict_proba_small_ok([PROMPT])),
        "heuristic_single": bench(lambda: heur.predict_proba_small_ok([PROMPT])),
    }

    batch = [PROMPT] * args.batch
    learned.predict_proba_small_ok(batch)
    t0 = time.perf_counter()
    learned.predict_proba_small_ok(batch)
    total_ms = (time.perf_counter() - t0) * 1000
    out["learned_batched"] = {
        "batch_size": args.batch,
        "total_ms": round(total_ms, 2),
        "per_item_ms": round(total_ms / args.batch, 4),
    }

    (cfg.paths.reports / "router_latency.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
