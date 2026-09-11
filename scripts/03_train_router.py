"""Train the router on sufficiency labels and report discrimination.

    python scripts/03_train_router.py --config configs/default.yaml

The split is grouped by task source so a router cannot pass by memorising a
benchmark's surface form... except that with only a handful of sources a group
split leaves too few groups to train on. Both splits are computed and both are
reported. The stratified number is the optimistic one; the grouped number is
the one to quote when asked whether this transfers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmrouter.config import Config  # noqa: E402
from llmrouter.router import (  # noqa: E402
    HeuristicRouter,
    LearnedRouter,
    build_pipeline,
    save_metadata,
)
from llmrouter.simulate import labels, read_records  # noqa: E402


def metrics(y, p) -> dict:
    if len(set(y.tolist())) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "brier": float("nan")}
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "ap": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.paths.ensure()
    records = read_records(cfg.paths.data / "records.jsonl")
    X = [r.prompt for r in records]
    y = labels(records, margin=cfg.sufficiency_margin)
    groups = [r.source for r in records]

    print(f"n={len(X)}  positives(small_ok)={y.mean():.3f}")
    if y.mean() in (0.0, 1.0):
        raise SystemExit(
            "Degenerate labels: one tier won every item. A router has nothing "
            "to learn here. Widen the task mix or the model gap."
        )

    idx = np.arange(len(X))
    tr, te = train_test_split(
        idx, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    pipe = build_pipeline(calibrate=not args.no_calibrate, seed=cfg.seed)
    pipe.fit([X[i] for i in tr], y[tr])
    learned = LearnedRouter(pipeline=pipe)

    p_te = learned.predict_proba_small_ok([X[i] for i in te])
    p_heur_te = HeuristicRouter().predict_proba_small_ok([X[i] for i in te])

    report = {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "label_rate": round(float(y.mean()), 4),
        "stratified_split": {
            "learned": metrics(y[te], p_te),
            "heuristic": metrics(y[te], p_heur_te),
            "always_small_baseline_auc": 0.5,
        },
    }

    # Grouped split: train on some sources, test on unseen ones.
    if len(set(groups)) >= 2:
        gtr, gte = next(
            GroupShuffleSplit(
                n_splits=1, test_size=0.34, random_state=cfg.seed
            ).split(X, y, groups)
        )
        gpipe = build_pipeline(calibrate=not args.no_calibrate, seed=cfg.seed)
        gpipe.fit([X[i] for i in gtr], y[gtr])
        gp = LearnedRouter(pipeline=gpipe).predict_proba_small_ok(
            [X[i] for i in gte]
        )
        report["grouped_split_unseen_sources"] = {
            "held_out_sources": sorted({groups[i] for i in gte}),
            "learned": metrics(y[gte], gp),
        }
    else:
        report["grouped_split_unseen_sources"] = "skipped: fewer than 2 sources"

    learned.save(cfg.paths.artifacts / "router.joblib")
    np.save(cfg.paths.artifacts / "test_index.npy", te)
    save_metadata(cfg.paths.artifacts / "router_meta.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
