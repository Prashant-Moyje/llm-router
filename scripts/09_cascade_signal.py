"""Cascade signal analysis. No API calls — uses cached records only.

    python scripts/09_cascade_signal.py --config configs/groq.yaml

Prompt-only routing failed because the input does not reveal whether the small
model will succeed. This asks a different question: does the small model's own
BEHAVIOUR reveal it?

Signals tested here are already stored on every record and cost nothing:

  * small_output_tokens - a reasoning model that thinks longer may be
    signalling difficulty. Free, and available before the large model is called.
  * small_latency_s     - largely a proxy for output length; included to check
    whether it adds anything independent.
  * prompt features + output tokens - does behaviour add to surface form?

Cascade economics are charged honestly: an escalated item pays BOTH tiers'
cost and BOTH latencies, because the small model already ran. A cascade that
escalates everything is strictly worse than always-large.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmrouter.config import Config  # noqa: E402
from llmrouter.features import extract  # noqa: E402
from llmrouter.simulate import (  # noqa: E402
    always_large,
    always_small,
    evaluate,
    labels,
    oracle,
    random_at_rate,
    read_records,
)


def auc(y, s) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    a = roc_auc_score(y, s)
    # A signal that is informative in reverse is still informative; report the
    # oriented value and note the direction separately.
    return float(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/groq.yaml")
    ap.add_argument("--records", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    path = Path(args.records) if args.records else cfg.paths.data / "records.jsonl"
    records = read_records(path)
    y = labels(records, margin=cfg.sufficiency_margin)
    n = len(records)

    print(f"n={n}  label_rate(small_ok)={y.mean():.4f}\n")

    tok = np.array([r.small_output_tokens for r in records], dtype=float)
    lat = np.array([r.small_latency_s for r in records], dtype=float)

    print("Signal separation (mean value by outcome):")
    for name, v in (("small_output_tokens", tok), ("small_latency_s", lat)):
        m1, m0 = v[y == 1].mean(), v[y == 0].mean()
        print(f"  {name:22s} small_ok={m1:8.1f}   small_failed={m0:8.1f}   "
              f"ratio={m0 / max(m1, 1e-9):.2f}x")

    print("\nUnivariate AUC for predicting small_ok (0.5 = useless):")
    results = {}
    for name, v in (("small_output_tokens", tok), ("small_latency_s", lat)):
        # Longer output is expected to indicate DIFFICULTY, so negate.
        a = auc(y, -v)
        results[name] = round(a, 4)
        print(f"  {name:22s} AUC={a:.4f}")

    # Multivariate: prompt features alone, behaviour alone, and both.
    X_prompt = np.array([extract(r.prompt) for r in records])
    X_behav = np.column_stack([tok, lat, np.log1p(tok)])
    X_both = np.column_stack([X_prompt, X_behav])

    idx = np.arange(n)
    tr, te = train_test_split(
        idx, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    print("\nHeld-out AUC (logistic regression):")
    for name, X in (
        ("prompt features only", X_prompt),
        ("small-model behaviour only", X_behav),
        ("prompt + behaviour", X_both),
    ):
        pipe = Pipeline([
            ("s", StandardScaler()),
            ("c", LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=cfg.seed)),
        ])
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        a = auc(y[te], p)
        results[name] = round(a, 4)
        print(f"  {name:28s} AUC={a:.4f}")

    # ------------------------------------------------------------------ #
    # Cascade economics on the held-out split
    # ------------------------------------------------------------------ #
    test = [records[i] for i in te]
    small = evaluate(test, always_small(test), "always_small", seed=cfg.seed)
    large = evaluate(test, always_large(test), "always_large", seed=cfg.seed)
    orc = evaluate(test, oracle(test), "oracle", seed=cfg.seed)

    print("\nReference policies (held-out):")
    for r in (small, large, orc):
        print(f"  {r.name:14s} q={r.quality:.4f}  ${r.cost_per_1k_usd:.4f}/1k  "
              f"p50={r.latency_p50_s:.2f}s")

    # Cascade using the best available behaviour signal, swept over thresholds.
    pipe = Pipeline([
        ("s", StandardScaler()),
        ("c", LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=cfg.seed)),
    ])
    pipe.fit(X_behav[tr], y[tr])
    p_te = pipe.predict_proba(X_behav[te])[:, 1]

    print("\nCascade frontier (escalated items pay BOTH tiers):")
    print(f"  {'tau':>5} {'keep_small':>11} {'quality':>8} {'$/1k':>9} "
          f"{'p50':>7}  vs_random")
    rows = []
    for tau in np.linspace(0.05, 0.95, 19):
        dec = p_te >= tau
        res = evaluate(test, dec, f"cascade@{tau:.2f}", cascade=True,
                       n_boot=200, seed=cfg.seed)
        rate = float(dec.mean())
        rnd = np.array([
            evaluate(test, random_at_rate(test, rate, seed=cfg.seed + s), "r",
                     cascade=True, n_boot=2).quality
            for s in range(60)
        ])
        beat = float((res.quality > rnd).mean())
        rows.append({
            "tau": round(float(tau), 3),
            "keep_small_share": round(rate, 4),
            "quality": round(res.quality, 4),
            "cost_per_1k_usd": round(res.cost_per_1k_usd, 4),
            "p50_s": round(res.latency_p50_s, 3),
            "beats_random_pct": round(beat, 3),
        })
        print(f"  {tau:5.2f} {rate:11.3f} {res.quality:8.4f} "
              f"{res.cost_per_1k_usd:9.4f} {res.latency_p50_s:7.2f}  {beat:.0%}")

    # Is any cascade point better than always-large on BOTH axes?
    wins = [
        r for r in rows
        if r["quality"] >= large.quality and r["cost_per_1k_usd"] < large.cost_per_1k_usd
    ]
    out = {
        "n": n,
        "auc": results,
        "reference": {
            "always_small": small.row(),
            "always_large": large.row(),
            "oracle": orc.row(),
        },
        "cascade_frontier": rows,
        "cascade_dominates_always_large": wins[:3],
    }
    dest = cfg.paths.reports / "cascade_signal.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    if wins:
        b = min(wins, key=lambda r: r["cost_per_1k_usd"])
        print(
            f"Cascade DOMINATES always-large at tau={b['tau']}: quality "
            f"{b['quality']} >= {large.quality:.4f} at "
            f"${b['cost_per_1k_usd']}/1k vs ${large.cost_per_1k_usd:.4f} "
            f"({1 - b['cost_per_1k_usd'] / large.cost_per_1k_usd:.1%} cheaper)."
        )
    else:
        print(
            "No cascade point matches always-large quality at lower cost. "
            "With double-billing on escalation and a small tier this cheap, "
            "the cascade has to escalate rarely to pay off - and it cannot "
            "identify which items to keep."
        )
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
