"""Recompute the summary from an existing records file. No API calls.

    python scripts/08_summarize.py --records data/records_C.jsonl --config configs/groq.yaml

Every number in reports/offline_eval.json is a pure function of the cached
records plus a price list, so a lost or overwritten summary is never a reason
to re-run models. Use --price-config to see the same measured run under a
different rate card (labelled COUNTERFACTUAL, as in 04_simulate.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmrouter.config import Config  # noqa: E402
from llmrouter.simulate import paired_test, read_records, reprice  # noqa: E402


def summarize(records, cfg, pricing_note: str) -> dict:
    n = len(records)
    if not n:
        raise SystemExit("No records found.")

    per_source = {}
    for src in sorted({r.source for r in records}):
        rows = [r for r in records if r.source == src]
        m = len(rows)
        per_source[src] = {
            "n": m,
            "small_acc": round(sum(r.small_score for r in rows) / m, 3),
            "large_acc": round(sum(r.large_score for r in rows) / m, 3),
            "label_rate": round(sum(r.small_ok for r in rows) / m, 3),
        }

    def mean(f):
        return sum(f(r) for r in records) / n

    tokens = {
        "small_mean_output_tokens": round(mean(lambda r: r.small_output_tokens), 1),
        "large_mean_output_tokens": round(mean(lambda r: r.large_output_tokens), 1),
        "rate_card_output_ratio": round(
            cfg.large.output_usd_per_mtok / max(cfg.small.output_usd_per_mtok, 1e-9), 2
        ),
        "effective_cost_ratio": round(
            sum(r.large_cost_usd for r in records)
            / max(sum(r.small_cost_usd for r in records), 1e-12),
            2,
        ),
    }

    return {
        "n_records": n,
        "pricing": pricing_note,
        "small_accuracy": round(sum(r.small_score for r in records) / n, 4),
        "large_accuracy": round(sum(r.large_score for r in records) / n, 4),
        "label_rate_small_ok": round(sum(r.small_ok for r in records) / n, 4),
        "small_cost_usd_total": round(sum(r.small_cost_usd for r in records), 4),
        "large_cost_usd_total": round(sum(r.large_cost_usd for r in records), 4),
        "models": f"{cfg.small.model_id} vs {cfg.large.model_id}",
        "tokens": tokens,
        "paired_test": paired_test(records),
        "per_source": per_source,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--config", default="configs/groq.yaml")
    ap.add_argument("--price-config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    records = read_records(Path(args.records))
    note = f"as measured ({cfg.small.model_id} / {cfg.large.model_id})"

    if args.price_config:
        pcfg = Config.load(args.price_config)
        records = reprice(records, pcfg.small, pcfg.large)
        note = (
            f"COUNTERFACTUAL: measured on {cfg.small.model_id}/"
            f"{cfg.large.model_id}, priced at {pcfg.small.model_id}/"
            f"{pcfg.large.model_id}"
        )

    summary = summarize(records, cfg, note)
    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    pt = summary["paired_test"]
    if not pt["significant_at_05"]:
        print(
            f"\nNo detectable quality gap: {pt['discordant_pairs']} discordant "
            f"pairs ({pt['small_only_correct']} small-only, "
            f"{pt['large_only_correct']} large-only), exact p="
            f"{pt['mcnemar_exact_p']}."
        )


if __name__ == "__main__":
    main()
