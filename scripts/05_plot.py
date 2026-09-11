"""Plot the cost/quality frontier: learned router vs heuristic vs reference points.

    python scripts/05_plot.py

The heuristic curve is on the same axes on purpose. A frontier plotted alone
always looks impressive; plotted against a six-line rule it either shows a real
gap or it does not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmrouter.config import Config  # noqa: E402
from llmrouter.simulate import (  # noqa: E402
    always_large,
    always_small,
    evaluate,
    oracle,
    random_at_rate,
    read_records,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = Config.load(args.config)

    records = read_records(cfg.paths.data / "records.jsonl")
    te = np.load(cfg.paths.artifacts / "test_index.npy")
    test = [records[i] for i in te]

    learned = pd.read_csv(cfg.paths.reports / "pareto_learned.csv")
    heur = pd.read_csv(cfg.paths.reports / "pareto_heuristic.csv")

    small = evaluate(test, always_small(test), "s", n_boot=200, seed=cfg.seed)
    large = evaluate(test, always_large(test), "l", n_boot=200, seed=cfg.seed)
    orc = evaluate(test, oracle(test), "o", n_boot=200, seed=cfg.seed)

    rnd = [
        evaluate(test, random_at_rate(test, r, seed=cfg.seed), "r",
                 n_boot=200, seed=cfg.seed)
        for r in np.linspace(0, 1, 21)
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(learned["cost_per_1k_usd"], learned["quality"], "-o", ms=3,
            lw=1.8, label="learned router")
    ax.plot(heur["cost_per_1k_usd"], heur["quality"], "--s", ms=3,
            lw=1.4, alpha=0.8, label="heuristic router")
    ax.plot([r.cost_per_1k_usd for r in rnd], [r.quality for r in rnd],
            ":", lw=1.4, color="grey", label="random at same rate")

    for res, marker, label in (
        (small, "v", f"always {cfg.small.name}"),
        (large, "^", f"always {cfg.large.name}"),
        (orc, "*", "oracle (upper bound)"),
    ):
        ax.scatter(res.cost_per_1k_usd, res.quality, marker=marker, s=140,
                   zorder=5, label=label)

    ax.set_xlabel("cost, USD per 1,000 requests")
    ax.set_ylabel("accuracy on held-out test split")
    ax.set_title("Cost/quality frontier")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out = cfg.paths.reports / "pareto.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
