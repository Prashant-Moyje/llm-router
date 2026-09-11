"""Offline replay: evaluate any routing policy without new API calls.

Both tiers are run once over the whole benchmark and every outcome - score,
token usage, cost, latency - is cached to disk. After that, *any* policy is
evaluated by replaying the cached record for whichever model it would have
chosen. Sweeping 200 thresholds costs nothing and is exactly reproducible.

Calling the API inside the threshold sweep instead would make each sweep cost
real money, take hours, and produce a different answer every run because of
sampling noise. Separating the expensive measurement from the cheap policy
search is the single most important structural decision in this repo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Record:
    """One benchmark item, measured on both tiers."""

    task_id: str
    prompt: str
    source: str
    small_score: float
    large_score: float
    small_cost_usd: float
    large_cost_usd: float
    small_latency_s: float
    large_latency_s: float
    # Token counts are stored so cost can be RECOMPUTED under different price
    # assumptions without re-running the models. Storing only the dollar figure
    # welds the records to one provider's price list; when that list is a poor
    # fit (a provider whose tiers are near-identically priced) the whole cost
    # axis becomes uninformative and the only remedy is paying for a new run.
    small_input_tokens: int = 0
    small_output_tokens: int = 0
    large_input_tokens: int = 0
    large_output_tokens: int = 0
    small_error: str | None = None
    large_error: str | None = None

    @property
    def small_ok(self) -> int:
        return int(self.small_score >= self.large_score)


def write_records(records: list[Record], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r)) + "\n")


def read_records(path: Path) -> list[Record]:
    with Path(path).open(encoding="utf-8") as fh:
        return [Record(**json.loads(line)) for line in fh if line.strip()]


def paired_test(records: list[Record]) -> dict:
    """McNemar's test on small vs large, plus the minimum detectable effect.

    The two tiers are scored on the SAME items, so the outcomes are paired and
    comparing two independent confidence intervals throws away most of the
    statistical power. Items where both tiers agree carry no information about
    which is better; only the discordant pairs do. McNemar uses exactly those.

    This exists because "large scored 1.7 points higher" is not a finding
    unless the discordant counts support it. Reporting an accuracy delta
    without a paired test on n=60 is how a coin flip becomes a headline.
    """
    from scipy import stats

    both = sum(1 for r in records if r.small_score and r.large_score)
    neither = sum(
        1 for r in records if not r.small_score and not r.large_score
    )
    small_only = sum(
        1 for r in records if r.small_score and not r.large_score
    )
    large_only = sum(
        1 for r in records if not r.small_score and r.large_score
    )

    n_disc = small_only + large_only
    # Exact binomial on the discordant pairs (correct at small n; the
    # chi-square approximation is unreliable below ~25 discordant pairs).
    p_value = (
        float(stats.binomtest(large_only, n_disc, 0.5).pvalue)
        if n_disc > 0
        else 1.0
    )

    return {
        "both_correct": both,
        "both_wrong": neither,
        "small_only_correct": small_only,
        "large_only_correct": large_only,
        "discordant_pairs": n_disc,
        "mcnemar_exact_p": round(p_value, 4),
        "significant_at_05": bool(p_value < 0.05),
        "note": (
            "Only discordant pairs carry signal. With few of them, no accuracy "
            "difference between the tiers is detectable at any sample size you "
            "can afford - which is itself the result."
        ),
    }


def reprice(records: list[Record], small_spec, large_spec) -> list[Record]:
    """Recompute costs from stored token counts under a different price list.

    This is a COUNTERFACTUAL, and must be labelled as one wherever it is
    reported: the model outputs, scores and latencies are whatever was actually
    measured, but the dollars answer "what would this same routing behaviour
    have cost at these other rates?"

    It is legitimate for exactly one thing - showing that the savings headline
    is a function of the price ratio rather than of the router (FAILURE_MODES
    §3) - and it is illegitimate as a way to quote a bigger number. Token counts
    are NOT transferable across model families: different tokenizers produce
    different counts for identical text, so repricing Llama token counts at
    Claude rates carries that error. Treat the result as an order-of-magnitude
    sensitivity check, never as a measurement.
    """
    from dataclasses import replace

    if all(r.small_input_tokens == 0 and r.large_input_tokens == 0 for r in records):
        raise ValueError(
            "Records carry no token counts, so cost cannot be recomputed. "
            "These were written by an older version of 02_run_offline_eval.py "
            "- re-run it to regenerate records.jsonl."
        )

    out = []
    for r in records:
        out.append(
            replace(
                r,
                small_cost_usd=small_spec.cost_usd(
                    r.small_input_tokens, r.small_output_tokens
                ),
                large_cost_usd=large_spec.cost_usd(
                    r.large_input_tokens, r.large_output_tokens
                ),
            )
        )
    return out


def labels(records: list[Record], margin: float = 0.0) -> np.ndarray:
    """y = 1 when the small model was good enough on this item."""
    return np.asarray(
        [int(r.small_score + margin >= r.large_score) for r in records], dtype=int
    )


# --------------------------------------------------------------------------- #
# Policy evaluation
# --------------------------------------------------------------------------- #


@dataclass
class PolicyResult:
    name: str
    quality: float
    cost_per_1k_usd: float
    latency_p50_s: float
    latency_p95_s: float
    small_share: float
    quality_ci95: tuple[float, float] = (0.0, 0.0)
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {
            "policy": self.name,
            "quality": round(self.quality, 4),
            "q_lo95": round(self.quality_ci95[0], 4),
            "q_hi95": round(self.quality_ci95[1], 4),
            "cost_per_1k_usd": round(self.cost_per_1k_usd, 4),
            "p50_s": round(self.latency_p50_s, 3),
            "p95_s": round(self.latency_p95_s, 3),
            "small_share": round(self.small_share, 4),
            **self.extra,
        }


def evaluate(
    records: list[Record],
    to_small: np.ndarray,
    name: str,
    cascade: bool = False,
    n_boot: int = 2000,
    seed: int = 13,
) -> PolicyResult:
    """Score a routing decision vector against cached outcomes.

    ``cascade=True`` means the small model was run first and then re-run on the
    large tier when escalated, so escalated items pay *both* bills and *both*
    latencies. Charging only the large-model price for a cascade is the most
    common way these systems get oversold.
    """
    to_small = np.asarray(to_small, dtype=bool)
    scores, costs, lats = [], [], []

    for r, s in zip(records, to_small, strict=True):
        if s:
            scores.append(r.small_score)
            costs.append(r.small_cost_usd)
            lats.append(r.small_latency_s)
        elif cascade:
            scores.append(r.large_score)
            costs.append(r.small_cost_usd + r.large_cost_usd)
            lats.append(r.small_latency_s + r.large_latency_s)
        else:
            scores.append(r.large_score)
            costs.append(r.large_cost_usd)
            lats.append(r.large_latency_s)

    scores_a = np.asarray(scores, dtype=float)
    costs_a = np.asarray(costs, dtype=float)
    lats_a = np.asarray(lats, dtype=float)

    rng = np.random.default_rng(seed)
    n = len(scores_a)
    boot = scores_a[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)

    return PolicyResult(
        name=name,
        quality=float(scores_a.mean()),
        cost_per_1k_usd=float(costs_a.mean() * 1000),
        latency_p50_s=float(np.percentile(lats_a, 50)),
        latency_p95_s=float(np.percentile(lats_a, 95)),
        small_share=float(to_small.mean()),
        quality_ci95=(
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ),
    )


# --------------------------------------------------------------------------- #
# Reference policies
# --------------------------------------------------------------------------- #


def always_small(records) -> np.ndarray:
    return np.ones(len(records), dtype=bool)


def always_large(records) -> np.ndarray:
    return np.zeros(len(records), dtype=bool)


def oracle(records: list[Record]) -> np.ndarray:
    """Upper bound: a router with perfect foresight.

    Not achievable. It is here to bound the headroom - if oracle saves 60% of
    spend, a real router claiming 58% deserves a second look at leakage.
    """
    return np.asarray([bool(r.small_ok) for r in records])


def random_at_rate(records, rate: float, seed: int = 13) -> np.ndarray:
    """Route a fixed random share to the small model.

    This is the baseline that actually matters. Any policy sending X% of
    traffic to a cheaper model saves money; the only question is whether it
    picks *better than chance* which X%. A learned router that matches random
    at the same escalation rate has learned nothing, however good its Pareto
    curve looks in isolation.
    """
    rng = np.random.default_rng(seed)
    return rng.random(len(records)) < rate


def threshold(probs: np.ndarray, tau: float) -> np.ndarray:
    return np.asarray(probs) >= tau


# --------------------------------------------------------------------------- #
# Frontier
# --------------------------------------------------------------------------- #


def sweep(
    records: list[Record],
    probs: np.ndarray,
    taus: np.ndarray | None = None,
    cascade: bool = False,
    seed: int = 13,
) -> list[dict]:
    if taus is None:
        taus = np.linspace(0.0, 1.0, 101)
    rows = []
    for tau in taus:
        res = evaluate(
            records,
            threshold(probs, tau),
            name=f"tau={tau:.2f}",
            cascade=cascade,
            n_boot=200,
            seed=seed,
        )
        row = res.row()
        row["tau"] = round(float(tau), 4)
        rows.append(row)
    return rows


def pick_tau_for_quality_floor(
    sweep_rows: list[dict], floor: float
) -> dict | None:
    """Cheapest threshold whose quality still clears ``floor``.

    This is the operating-point question a team actually asks: "I will accept a
    two-point accuracy loss - what does that save me?" Answering it needs a
    stated budget, not an argmax of some blended score.
    """
    ok = [r for r in sweep_rows if r["quality"] >= floor]
    return min(ok, key=lambda r: r["cost_per_1k_usd"]) if ok else None


def summarize(
    records: list[Record], probs: np.ndarray, seed: int = 13
) -> list[PolicyResult]:
    small = evaluate(records, always_small(records), "always_small", seed=seed)
    large = evaluate(records, always_large(records), "always_large", seed=seed)
    orc = evaluate(records, oracle(records), "oracle", seed=seed)
    results = [small, large, orc]

    for tau in (0.3, 0.5, 0.7):
        dec = threshold(probs, tau)
        results.append(evaluate(records, dec, f"router@tau={tau}", seed=seed))
        rate = float(dec.mean())
        results.append(
            evaluate(
                records,
                random_at_rate(records, rate, seed=seed),
                f"random@rate={rate:.2f}",
                seed=seed,
            )
        )
    return results


def retention(policy_q: float, small_q: float, large_q: float) -> float:
    """Share of the small->large quality gap the policy recovers.

    Reported instead of raw accuracy because raw accuracy hides the size of the
    gap. Retaining 97% of a 3-point gap is a rounding error; retaining 97% of a
    30-point gap is the whole result.
    """
    denom = large_q - small_q
    return float("nan") if abs(denom) < 1e-9 else (policy_q - small_q) / denom
