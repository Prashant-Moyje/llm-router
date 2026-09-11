"""Sweep thresholds on held-out data and write the cost/quality report.

    python scripts/04_simulate.py --quality-floor-drop 0.02

Everything here is evaluated on the test split only. Selecting tau on the same
data used to train the router would report a threshold tuned to noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmrouter.config import Config  # noqa: E402
from llmrouter.router import HeuristicRouter, LearnedRouter  # noqa: E402
from llmrouter.simulate import (  # noqa: E402
    always_large,
    always_small,
    evaluate,
    oracle,
    random_at_rate,
    pick_tau_for_quality_floor,
    read_records,
    reprice,
    retention,
    summarize,
    sweep,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--quality-floor-drop",
        type=float,
        default=0.02,
        help="Accuracy loss you are willing to accept vs always-large.",
    )
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument(
        "--price-config",
        default=None,
        help=(
            "Recompute costs from stored token counts using another config's "
            "prices. COUNTERFACTUAL - quality and latency stay as measured, "
            "dollars become 'what this would have cost at those rates'. "
            "Label it as such in any report."
        ),
    )
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.paths.ensure()

    records = read_records(cfg.paths.data / "records.jsonl")

    price_note = f"measured ({cfg.small.model_id} / {cfg.large.model_id})"
    if args.price_config:
        pcfg = Config.load(args.price_config)
        records = reprice(records, pcfg.small, pcfg.large)
        price_note = (
            f"COUNTERFACTUAL: token counts measured on "
            f"{cfg.small.model_id}/{cfg.large.model_id}, priced at "
            f"{pcfg.small.model_id}/{pcfg.large.model_id} rates"
        )
        print(f"!! {price_note}\n")

    te = np.load(cfg.paths.artifacts / "test_index.npy")
    test = [records[i] for i in te]

    router = LearnedRouter.load(cfg.paths.artifacts / "router.joblib")
    prompts = [r.prompt for r in test]
    probs = router.predict_proba_small_ok(prompts)
    probs_heur = HeuristicRouter().predict_proba_small_ok(prompts)

    small = evaluate(test, always_small(test), "always_small", seed=cfg.seed)
    large = evaluate(test, always_large(test), "always_large", seed=cfg.seed)
    orc = evaluate(test, oracle(test), "oracle", seed=cfg.seed)

    table = pd.DataFrame([r.row() for r in summarize(test, probs, seed=cfg.seed)])
    # Use the unrounded policy quality here. Feeding the 4-dp display value in
    # made always_small report retention 0.0008 instead of exactly 0.
    table["quality_retention"] = [
        round(retention(q, small.quality, large.quality), 4)
        for q in (r.quality for r in summarize(test, probs, seed=cfg.seed))
    ]
    table["cost_vs_large"] = (
        table["cost_per_1k_usd"] / max(large.cost_per_1k_usd, 1e-12)
    ).round(4)

    rows = sweep(test, probs, cascade=args.cascade, seed=cfg.seed)
    rows_heur = sweep(test, probs_heur, cascade=args.cascade, seed=cfg.seed)
    pd.DataFrame(rows).to_csv(cfg.paths.reports / "pareto_learned.csv", index=False)
    pd.DataFrame(rows_heur).to_csv(
        cfg.paths.reports / "pareto_heuristic.csv", index=False
    )

    floor = large.quality - args.quality_floor_drop
    pick = pick_tau_for_quality_floor(rows, floor)
    pick_h = pick_tau_for_quality_floor(rows_heur, floor)

    headline = {
        "pricing": price_note,
        "quality_floor": round(floor, 4),
        "always_small": small.row(),
        "always_large": large.row(),
        "chosen_learned": pick,
        "chosen_heuristic": pick_h,
    }
    if pick:
        headline["cost_reduction_vs_large"] = round(
            1 - pick["cost_per_1k_usd"] / max(large.cost_per_1k_usd, 1e-12), 4
        )
        headline["p95_latency_reduction_vs_large"] = round(
            1 - pick["p95_s"] / max(large.latency_p95_s, 1e-12), 4
        )
        headline["quality_retention"] = round(
            retention(pick["quality"], small.quality, large.quality), 4
        )
        headline["quality_drop_ci_overlaps_large"] = bool(
            pick["q_hi95"] >= large.quality_ci95[0]
        )

    # THE comparison that decides whether the router earned anything. A policy
    # sending X% of traffic to a cheaper tier always saves money; the only
    # question is whether it picks better than chance WHICH X%. The fixed-tau
    # rows above can all collapse to one escalation rate, in which case this is
    # the only place the question gets asked.
    if pick:
        rate = pick["small_share"]
        rnd_qs = []
        for sd in range(200):
            dec = random_at_rate(test, rate, seed=cfg.seed + sd)
            rnd_qs.append(
                evaluate(test, dec, "r", n_boot=2, seed=cfg.seed).quality
            )
        rnd_qs = np.asarray(rnd_qs)
        delta = pick["quality"] - float(rnd_qs.mean())
        oracle_gap = orc.quality - float(rnd_qs.mean())
        headline["vs_random_at_matched_rate"] = {
            "matched_small_share": round(rate, 4),
            "random_quality_mean": round(float(rnd_qs.mean()), 4),
            "random_quality_p05_p95": [
                round(float(np.percentile(rnd_qs, 5)), 4),
                round(float(np.percentile(rnd_qs, 95)), 4),
            ],
            "router_quality": pick["quality"],
            "router_minus_random": round(delta, 4),
            "router_beats_random_in_pct_of_draws": round(
                float((pick["quality"] > rnd_qs).mean()), 3
            ),
            "share_of_random_to_oracle_gap_captured": (
                round(delta / oracle_gap, 3) if abs(oracle_gap) > 1e-9 else None
            ),
        }

    table.to_csv(cfg.paths.reports / "policy_comparison.csv", index=False)
    (cfg.paths.reports / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )

    print(table.to_string(index=False))
    print("\n" + json.dumps(headline, indent=2))
    if pick and headline.get("quality_drop_ci_overlaps_large"):
        print(
            "\nNote: the router's quality CI overlaps always-large. On this "
            "sample the quality loss is not statistically distinguishable "
            "from zero - report it that way, not as 'no quality loss'."
        )


if __name__ == "__main__":
    main()
