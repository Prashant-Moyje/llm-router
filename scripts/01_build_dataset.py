"""Build the benchmark task set.

    python scripts/01_build_dataset.py --sources gsm8k mmlu --limit 400
    python scripts/01_build_dataset.py --sources synthetic --limit 600
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmrouter.config import Config  # noqa: E402
from llmrouter.tasks import (  # noqa: E402
    load_gsm8k,
    load_mmlu,
    load_synthetic,
    write_tasks,
)

# Subjects chosen to sit at the capability boundary between a ~20B and a
# ~120B model. Easy subjects (elementary_mathematics, high_school_*) are
# excluded on purpose: when both tiers score ~1.0 the items yield degenerate
# labels, contribute nothing to training, and dilute the signal from the
# subjects that do discriminate. Check per_source in reports/offline_eval.json
# after a pilot run and drop any subject whose label_rate is ~0 or ~1.
# Revised after a 57-item pilot. At default reasoning effort, abstract_algebra,
# college_physics and formal_logic all came back at label_rate 1.0 - both tiers
# scored perfectly, so every item was a degenerate label. They are dropped.
#
# Retained because they discriminated: professional_law (0.60),
# moral_scenarios (0.78). Added as further candidates in the same difficulty
# band; re-check per_source after your own pilot and prune again. Subject
# difficulty is relative to the specific model pair, so this list is not
# portable to a different pair.
# Pruned across two pilots. Dropped for label_rate ~1.0 (no signal against
# this model pair): abstract_algebra, college_physics, formal_logic,
# elementary_mathematics, high_school_mathematics, econometrics,
# high_school_statistics. GSM8K dropped for the same reason.
#
# What survives is the low-accuracy end of MMLU, where BOTH tiers are near
# chance on 4-way multiple choice (0.25). That is its own problem: items
# neither tier can do produce label_rate 1.0 under the >= rule and teach the
# router to spot impossible items rather than easy ones (§10).
MMLU_SUBJECTS = (
    "professional_law",
    "moral_scenarios",
    "virology",
    "professional_accounting",
    "college_medicine",
    "human_aging",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sources", nargs="+", default=["synthetic"])
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.paths.ensure()

    tasks = []
    for src in args.sources:
        if src == "gsm8k":
            tasks += load_gsm8k(limit=args.limit)
        elif src == "mmlu":
            tasks += load_mmlu(limit=args.limit, subjects=MMLU_SUBJECTS)
        elif src == "synthetic":
            tasks += load_synthetic(n=args.limit, seed=cfg.seed)
        else:
            raise SystemExit(f"unknown source: {src}")

    out = Path(args.out) if args.out else cfg.paths.data / "tasks.jsonl"
    write_tasks(tasks, out)
    print(f"wrote {len(tasks)} tasks -> {out}")
    for k, v in sorted(Counter(t.source for t in tasks).items()):
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
