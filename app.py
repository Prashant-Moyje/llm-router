"""Gradio demo for Hugging Face Spaces.

Deliberately framed around the measurement, not the router. The router does not
work on this task (held-out AUC 0.589; chance on unseen subjects), so a demo
that showed a confident tier decision and nothing else would misrepresent the
result this project exists to report. Every routing decision is shown alongside
its reliability, and the frontier tab lets you see for yourself that the router
barely separates from random at a matched escalation rate.

Run locally:  python app.py
On Spaces:    set app_file: app.py in README.md front matter
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from llmrouter.config import Config  # noqa: E402
from llmrouter.router import HeuristicRouter, LearnedRouter  # noqa: E402

CONFIG = ROOT / "configs/groq.yaml"
MODEL = ROOT / "artifacts/router.joblib"
REPORTS = ROOT / "reports"

cfg = Config.load(CONFIG)
router = LearnedRouter.load(MODEL) if MODEL.exists() else None
heuristic = HeuristicRouter()


def _load(name: str, default=None):
    # router_meta.json is written to artifacts/, everything else to reports/.
    # Reading it from the wrong directory silently yields {} and renders the
    # AUC figures as "None" — which would quietly delete the demo's honesty.
    p = REPORTS / name
    if not p.exists():
        p = ROOT / "artifacts" / name
    if not p.exists():
        return default
    return (
        json.loads(p.read_text(encoding="utf-8"))
        if p.suffix == ".json"
        else pd.read_csv(p)
    )


HEADLINE = _load("headline.json", {})
PARETO = _load("pareto_learned.csv")
PARETO_H = _load("pareto_heuristic.csv")
META = _load("router_meta.json", {})
CASCADE = _load("cascade_signal.json", {})

REF = HEADLINE.get("always_small", {}), HEADLINE.get("always_large", {})
SMALL_REF, LARGE_REF = REF


# --------------------------------------------------------------------------- #
# Tab 1: routing decision
# --------------------------------------------------------------------------- #

EXAMPLES = [
    "What is the capital of France?",
    "A firm's marginal cost curve intersects average total cost at its minimum. "
    "If fixed costs rise by 20%, what happens to the output level at which ATC "
    "is minimised?",
    "Summarise this email in one sentence.",
    "Prove that the set of continuous functions on [0,1] with the sup norm is "
    "complete, and explain where completeness of the reals is used.",
]


def route(prompt: str, tau: float):
    if not prompt.strip():
        return "Enter a prompt.", "", ""
    if router is None:
        return (
            "**artifacts/router.joblib is missing.** Run "
            "`python scripts/03_train_router.py` and redeploy.",
            "",
            "",
        )

    t0 = time.perf_counter()
    p = float(router.predict_proba_small_ok([prompt])[0])
    ms = (time.perf_counter() - t0) * 1000
    p_heur = float(heuristic.predict_proba_small_ok([prompt])[0])

    small = p >= tau
    tier = cfg.small if small else cfg.large
    label = "SMALL" if small else "LARGE"

    decision = f"""### → {label} tier: `{tier.model_id}`

| | |
|---|---|
| P(small model sufficient) | **{p:.3f}** |
| threshold τ | {tau:.2f} |
| router latency | {ms:.2f} ms |
| heuristic baseline says | {p_heur:.3f} |
| tier price | ${tier.input_usd_per_mtok:.2f} in / ${tier.output_usd_per_mtok:.2f} out per MTok |
"""

    auc = META.get("stratified_split", {}).get("learned", {}).get("auc")
    auc_g = (
        META.get("grouped_split_unseen_sources", {})
        .get("learned", {})
        .get("auc")
    )
    vs_rand = HEADLINE.get("vs_random_at_matched_rate", {})
    beat = vs_rand.get("router_beats_random_in_pct_of_draws")
    beat = f"{beat:.1%}" if isinstance(beat, (int, float)) else "under 95%"
    caveat = f"""### How much to trust this

This number is **not reliable**, and that is the project's finding rather than
a caveat about the demo.

- Held-out AUC **{auc}** in-distribution (0.5 = chance)
- AUC **{auc_g}** on subjects the model never saw — **chance**
- A six-line heuristic scores **{META.get('stratified_split', {}).get('heuristic', {}).get('auc')}**, beating it
- At its operating point it beats random routing at the same escalation rate in
  only **{beat}** of draws — short of significance

The classifier was trained on MMLU items with ~40 negative examples. Reasoning
difficulty is largely not visible in a prompt's surface form: two near-identical
law questions differ in whether a 20B model answers them correctly.
"""
    return decision, caveat, ""


# --------------------------------------------------------------------------- #
# Tab 2: frontier explorer
# --------------------------------------------------------------------------- #


def frontier(tau: float):
    if PARETO is None:
        return "Run `python scripts/04_simulate.py` to generate the frontier.", None

    row = PARETO.iloc[(PARETO["tau"] - tau).abs().argsort().iloc[0]]
    sq = SMALL_REF.get("quality", 0.0)
    lq = LARGE_REF.get("quality", 1.0)
    lc = LARGE_REF.get("cost_per_1k_usd", 1.0)

    # Expected quality of random routing at this same escalation rate.
    share = float(row["small_share"])
    rand_q = share * sq + (1 - share) * lq
    retention = (row["quality"] - sq) / (lq - sq) if lq != sq else float("nan")

    md = f"""### τ = {row['tau']:.2f} → {share:.1%} of traffic to the small tier

| metric | router | random at same rate | always large |
|---|---|---|---|
| accuracy | **{row['quality']:.4f}** | {rand_q:.4f} | {lq:.4f} |
| cost per 1k requests | **${row['cost_per_1k_usd']:.4f}** | ${row['cost_per_1k_usd']:.4f} | ${lc:.4f} |
| p50 latency | {row['p50_s']:.2f}s | — | {LARGE_REF.get('p50_s', 0):.2f}s |
| p95 latency | {row['p95_s']:.2f}s | — | {LARGE_REF.get('p95_s', 0):.2f}s |

- cost vs always-large: **{row['cost_per_1k_usd'] / lc:.1%}**
- quality retention: **{retention:.1%}** of the small→large gap
- router advantage over random at this rate: **{row['quality'] - rand_q:+.4f}**

Random and the router spend the *same* amount at a given escalation rate — they
differ only in which requests they escalate. That difference is the entire value
of a router, and here it is small.
"""
    return md, _plot(tau)


def _plot(tau: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sq = SMALL_REF.get("quality", 0)
    lq = LARGE_REF.get("quality", 1)

    ax.plot(PARETO["cost_per_1k_usd"], PARETO["quality"], "-o", ms=3,
            label="learned router")
    if PARETO_H is not None:
        ax.plot(PARETO_H["cost_per_1k_usd"], PARETO_H["quality"], "--s", ms=3,
                alpha=0.7, label="heuristic")

    rand_q = PARETO["small_share"] * sq + (1 - PARETO["small_share"]) * lq
    ax.plot(PARETO["cost_per_1k_usd"], rand_q, ":", color="grey", lw=2,
            label="random at same rate")

    ax.scatter(SMALL_REF.get("cost_per_1k_usd", 0), sq, marker="v", s=130,
               zorder=5, label="always small")
    ax.scatter(LARGE_REF.get("cost_per_1k_usd", 0), lq, marker="^", s=130,
               zorder=5, label="always large")

    cur = PARETO.iloc[(PARETO["tau"] - tau).abs().argsort().iloc[0]]
    ax.scatter(cur["cost_per_1k_usd"], cur["quality"], marker="*", s=320,
               color="crimson", zorder=6, label=f"τ={cur['tau']:.2f}")

    ax.set_xlabel("cost, USD per 1,000 requests")
    ax.set_ylabel("accuracy (held-out, n=144)")
    ax.set_title("Cost/quality frontier — router vs random at matched rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

FINDINGS = """
## What this project measured

Route each request to a cheap or expensive model based on a prediction of
whether the cheap one suffices, then measure what that trades away.

**480 MMLU items, `openai/gpt-oss-20b` vs `openai/gpt-oss-120b` on Groq.**
Exact-match scoring, no LLM judge.

### 1. The headroom is real

An oracle router scores **0.799 accuracy at 27% of always-large cost** —
beating always-large outright, because 28 of 480 items are ones the small model
gets right and the large one gets wrong.

The tier gap is significant: 86 discordant pairs, 58 vs 28,
**McNemar exact p = 0.0016**. A 60-item pilot gave p = 1.00 — underpowered, not
flat, which is why the paired test matters.

### 2. But difficulty prediction fails from two directions

| signal | held-out AUC |
|---|---|
| prompt text | 0.558 (0.504 on unseen subjects — chance) |
| small model's output length | 0.549 |
| both combined | 0.589 |

Items the small model gets wrong *do* run longer — 129.4 vs 93.5 output tokens,
a real 1.38× separation. It is simply far too weak to route on.

### 3. Two cost findings independent of the router

**The cheap tier can cost more.** With both tiers at default reasoning effort,
the 20B emitted 359 output tokens to the 120B's 270 — effective cost ratio
**0.97**. The weaker model compensates with longer reasoning, and with output
priced 5× input that erased a 1.2× rate-card advantage. Invisible unless you
store realised token counts.

**Cascades double-bill.** An escalated request pays both tiers. At one threshold
the cascade matched always-large's accuracy while costing **7% more**.

---

*Full write-up: `RESULTS.md`. Limitations: `FAILURE_MODES.md`.*
"""

with gr.Blocks(title="LLM cost/latency router", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# LLM cost/latency router\n"
        "Routing by predicted task difficulty — and a measurement showing it "
        "**doesn't work** on this task. The oracle proves the headroom exists; "
        "neither the prompt nor the model's own output length can reach it."
    )

    with gr.Tab("Try the router"):
        gr.Markdown(
            "Returns a routing decision only — no model is called, so this is "
            "free and instant."
        )
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(label="Prompt", lines=5,
                                 placeholder="Ask anything…")
                tau_in = gr.Slider(0.0, 1.0, value=0.88, step=0.01,
                                   label="τ — route to small when P ≥ τ")
                btn = gr.Button("Route", variant="primary")
                gr.Examples(EXAMPLES, inputs=inp)
            with gr.Column(scale=2):
                out_dec = gr.Markdown()
        out_caveat = gr.Markdown()
        btn.click(route, [inp, tau_in], [out_dec, out_caveat, gr.State()])

    with gr.Tab("Cost/quality frontier"):
        gr.Markdown(
            "Every point replays the cached 480-item evaluation, so this costs "
            "nothing and is exactly reproducible. The grey line is random "
            "routing at the same escalation rate — the baseline that decides "
            "whether the router earned anything."
        )
        tau_f = gr.Slider(0.0, 1.0, value=0.88, step=0.01, label="τ")
        out_md = gr.Markdown()
        out_plot = gr.Plot()
        tau_f.change(frontier, tau_f, [out_md, out_plot])
        demo.load(frontier, tau_f, [out_md, out_plot])

    with gr.Tab("Findings"):
        gr.Markdown(FINDINGS)

if __name__ == "__main__":
    demo.launch()
