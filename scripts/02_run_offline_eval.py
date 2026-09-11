"""Run both tiers over every task once and cache the outcomes.

This is the only script that spends money. Everything downstream replays the
JSONL it writes.

    python scripts/02_run_offline_eval.py --provider mock
    python scripts/02_run_offline_eval.py --provider anthropic --workers 8

Resumable: existing task_ids in the output file are skipped, so a run killed
halfway does not have to be paid for twice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmrouter.config import Config  # noqa: E402
from llmrouter.providers import get_provider  # noqa: E402
from llmrouter.simulate import (  # noqa: E402
    Record,
    paired_test,
    read_records,
    write_records,
)
from llmrouter.tasks import read_tasks, score  # noqa: E402


def _stratified_sample(tasks, limit: int, seed: int):
    """Take `limit` tasks spread evenly over sources, deterministically."""
    import random

    rng = random.Random(seed)
    by_src = defaultdict(list)
    for t in tasks:
        by_src[t.source].append(t)
    for v in by_src.values():
        rng.shuffle(v)

    out, sources = [], sorted(by_src)
    i = 0
    while len(out) < limit and any(by_src[s] for s in sources):
        src = sources[i % len(sources)]
        if by_src[src]:
            out.append(by_src[src].pop())
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "anthropic", "groq", "openrouter", "openai_compat", "ollama"],
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help=(
            "Retry budget per call. High values look safer but collapse "
            "throughput against a SUSTAINED rate limit: every call walks the "
            "full backoff ladder. 4 is right for transient 429s; raise it only "
            "if errors are sporadic rather than constant."
        ),
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "0 = all tasks. Otherwise samples EVENLY ACROSS SOURCES, not the "
            "first N. Truncating a source-ordered file yields a single-source "
            "sample, which on an easy source produces degenerate labels and "
            "tells you nothing about the capability gap."
        ),
    )
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.paths.ensure()
    tasks = read_tasks(cfg.paths.data / "tasks.jsonl")
    if args.limit and args.limit < len(tasks):
        tasks = _stratified_sample(tasks, args.limit, cfg.seed)
        by_src = Counter(t.source for t in tasks)
        print(f"sampling {len(tasks)} tasks across {len(by_src)} sources:")
        for k, v in sorted(by_src.items()):
            print(f"  {k:32s} {v}")

    out_path = cfg.paths.data / "records.jsonl"
    done: dict[str, Record] = {}
    if out_path.exists():
        done = {r.task_id: r for r in read_records(out_path)}
        print(f"resuming: {len(done)} records already cached")

    todo = [t for t in tasks if t.task_id not in done]
    provider = get_provider(args.provider, cfg)
    if hasattr(provider, "max_retries"):
        provider.max_retries = args.max_retries

    def run_one(task) -> Record:
        s = provider.generate(task.prompt, cfg.small)
        b = provider.generate(task.prompt, cfg.large)
        return Record(
            task_id=task.task_id,
            prompt=task.prompt,
            source=task.source,
            small_score=score(task, s.text),
            large_score=score(task, b.text),
            small_cost_usd=s.cost_usd,
            large_cost_usd=b.cost_usd,
            small_latency_s=s.latency_s,
            large_latency_s=b.latency_s,
            small_input_tokens=s.input_tokens,
            small_output_tokens=s.output_tokens,
            large_input_tokens=b.input_tokens,
            large_output_tokens=b.output_tokens,
            small_error=s.error,
            large_error=b.error,
        )

    # Append each record the moment it lands. The previous version buffered
    # everything and wrote once at the end, so killing a long run - or a
    # crash, or a laptop sleeping - discarded every item already paid for.
    # On a rate-limited free tier a full pass can take hours, which makes
    # "all or nothing" the wrong durability model.
    new: list[Record] = []
    if todo:
        started = time.time()
        with out_path.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(
            max_workers=args.workers
        ) as pool:
            for i, rec in enumerate(pool.map(run_one, todo), 1):
                new.append(rec)
                sink.write(json.dumps(asdict(rec)) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
                if i % 10 == 0 or i == len(todo):
                    elapsed = time.time() - started
                    rate = i / max(elapsed, 1e-9)
                    eta_min = (len(todo) - i) / max(rate, 1e-9) / 60
                    errs = sum(
                        1 for r in new if r.small_error or r.large_error
                    )
                    print(
                        f"  {i}/{len(todo)}  {elapsed/60:.1f}m elapsed  "
                        f"ETA {eta_min:.0f}m  errors={errs}",
                        flush=True,
                    )

    # Appending means a task_id can appear twice across resumed runs; keep the
    # newest occurrence.
    merged = {r.task_id: r for r in list(done.values()) + new}
    records = list(merged.values())
    # Drop items where either tier errored; a failed call is not a zero score,
    # and silently scoring it 0 would attribute an infrastructure fault to
    # model quality.
    clean = [r for r in records if not r.small_error and not r.large_error]
    failed = [r for r in records if r.small_error or r.large_error]
    dropped = len(failed)

    # Persist the failures. Dropping them from the record set is correct for
    # scoring, but discarding the error text makes a total failure
    # undebuggable - you learn that 300 calls failed and nothing about why.
    if failed:
        fail_path = cfg.paths.reports / "failures.jsonl"
        with fail_path.open("w", encoding="utf-8") as fh:
            for r in failed:
                fh.write(json.dumps({
                    "task_id": r.task_id,
                    "small_error": r.small_error,
                    "large_error": r.large_error,
                }) + "\n")
        print(f"\n{dropped} item(s) failed. First 3 errors:")
        for r in failed[:3]:
            print(f"  {r.task_id}")
            if r.small_error:
                print(f"    small({cfg.small.model_id}): {r.small_error[:300]}")
            if r.large_error:
                print(f"    large({cfg.large.model_id}): {r.large_error[:300]}")
        print(f"  full list -> {fail_path}\n")

    write_records(clean, out_path)
    n = len(clean)

    # Per-source breakdown. The capability gap is rarely uniform: a source
    # where both tiers score identically contributes only degenerate labels and
    # dilutes training. This shows which sources carry signal.
    per_source = {}
    for src in sorted({r.source for r in clean}):
        rows = [r for r in clean if r.source == src]
        m = len(rows)
        per_source[src] = {
            "n": m,
            "small_acc": round(sum(r.small_score for r in rows) / m, 3),
            "large_acc": round(sum(r.large_score for r in rows) / m, 3),
            "label_rate": round(sum(r.small_ok for r in rows) / m, 3),
        }

    # Mean output tokens per tier. A smaller reasoning model often emits LONGER
    # chains to compensate for weaker capability, and with output priced well
    # above input this can erase most of the rate-card price advantage. The
    # effective ratio below is what actually drives savings.
    def _mean(f):
        return sum(f(r) for r in clean) / max(n, 1)

    tokens = {
        "small_mean_output_tokens": round(_mean(lambda r: r.small_output_tokens), 1),
        "large_mean_output_tokens": round(_mean(lambda r: r.large_output_tokens), 1),
        "rate_card_output_ratio": round(
            cfg.large.output_usd_per_mtok / max(cfg.small.output_usd_per_mtok, 1e-9), 2
        ),
        "effective_cost_ratio": round(
            sum(r.large_cost_usd for r in clean)
            / max(sum(r.small_cost_usd for r in clean), 1e-12),
            2,
        ),
    }

    summary = {
        "n_records": n,
        "dropped_errors": dropped,
        "small_accuracy": round(sum(r.small_score for r in clean) / max(n, 1), 4),
        "large_accuracy": round(sum(r.large_score for r in clean) / max(n, 1), 4),
        "label_rate_small_ok": round(sum(r.small_ok for r in clean) / max(n, 1), 4),
        "small_cost_usd_total": round(sum(r.small_cost_usd for r in clean), 4),
        "large_cost_usd_total": round(sum(r.large_cost_usd for r in clean), 4),
        "provider": args.provider,
        "pricing_basis": f"{cfg.small.pricing_basis}/{cfg.large.pricing_basis}",
        "models": f"{cfg.small.model_id} vs {cfg.large.model_id}",
        "tokens": tokens,
        "paired_test": paired_test(clean) if clean else {},
        "per_source": per_source,
    }
    (cfg.paths.reports / "offline_eval.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    pt = summary.get("paired_test", {})
    if pt and not pt.get("significant_at_05"):
        print(
            f"\nNO DETECTABLE QUALITY GAP: small={summary['small_accuracy']} "
            f"large={summary['large_accuracy']}, but only "
            f"{pt['discordant_pairs']} discordant pairs "
            f"({pt['small_only_correct']} small-only, "
            f"{pt['large_only_correct']} large-only), exact p="
            f"{pt['mcnemar_exact_p']}. The accuracy difference is not "
            "distinguishable from chance. If this holds at full n, the "
            "correct policy is always-small and a router adds nothing."
        )

    lr = summary["label_rate_small_ok"]
    if n and (lr >= 0.98 or lr <= 0.02):
        print(
            "\nDEGENERATE LABELS: one tier won essentially every item "
            f"(label_rate={lr}). The router has nothing to learn and step 3 "
            "will refuse to train. Add harder sources, or widen the model gap."
        )
    if tokens["effective_cost_ratio"] < 1.5:
        print(
            f"\nNARROW EFFECTIVE COST GAP: {tokens['effective_cost_ratio']}x "
            f"(rate card implies {tokens['rate_card_output_ratio']}x on output). "
            "The small tier is emitting "
            f"{tokens['small_mean_output_tokens']} output tokens vs "
            f"{tokens['large_mean_output_tokens']} for the large one. Savings "
            "will be capped by this, not by routing accuracy."
        )
    if args.provider == "mock":
        print("\n[mock provider] numbers above are synthetic. Do not report them.")


if __name__ == "__main__":
    main()
