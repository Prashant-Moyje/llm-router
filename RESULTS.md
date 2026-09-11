# Results

**Headline: prompt-only routing does not work on this task.** The measurement
apparatus is sound and the cost/quality headroom is real, but a learned
classifier over prompt text cannot predict which items the cheap model will
handle. This is reported as the primary finding rather than buried.

Setup: 480 MMLU items across 6 subjects (professional_law, moral_scenarios,
virology, professional_accounting, econometrics, high_school_statistics),
`openai/gpt-oss-20b` at `reasoning_effort: low` vs `openai/gpt-oss-120b` at
`high`, served by Groq. Scoring is exact match on a normalised final answer;
no LLM judge. Dollar figures are Groq's published rates applied to measured
token counts (`pricing_basis: imputed`) — free-tier inference is billed at $0,
so these model what the traffic would cost, they are not a bill.

## 1. There is a real quality gap, and real headroom

| policy | accuracy | 95% CI | $/1k req | p50 | p95 | % to small |
|---|---|---|---|---|---|---|
| always small (20B) | 0.681 | 0.604–0.757 | 0.074 | 0.55s | 0.91s | 100% |
| always large (120B) | 0.736 | 0.660–0.806 | 0.554 | 2.03s | 4.93s | 0% |
| **oracle** | **0.799** | 0.729–0.861 | **0.147** | 0.57s | 3.31s | 88% |

Held-out test split, n=144.

The oracle is the interesting row: **it beats always-large on accuracy while
spending 27% as much.** Routing here is not purely a quality-for-cost trade,
because 28 of 480 items are ones the small model gets right and the large model
gets wrong. A perfect router would dominate the expensive model outright.

The tier gap is significant. Outcomes are paired on identical items, so only
discordant pairs carry information:

|  | large correct | large wrong |
|---|---|---|
| **small correct** | 305 | 28 |
| **small wrong** | 58 | 89 |

86 discordant pairs, 58 vs 28, **McNemar exact p = 0.0016**.

A 60-item pilot of the same configuration gave 5 vs 4 discordant pairs and
p = 1.00. That pilot was underpowered, not flat — which is precisely why the
paired test is computed rather than comparing two independent confidence
intervals.

## 2. The learned router does not beat its baselines

| measure | learned | heuristic | chance |
|---|---|---|---|
| AUC, stratified split | 0.593 | **0.658** | 0.500 |
| AUC, held-out subjects | **0.504** | — | 0.500 |
| Brier | 0.103 | 0.654 | — |

Three independent readings agree. On unseen subjects the learned router is at
chance. In-distribution it is beaten by a six-line hand-written heuristic. Its
only genuine win is calibration (Brier 0.103 vs 0.654), which makes `tau`
interpretable as a probability but does not make its ranking useful.

**Against the baseline that matters.** At its chosen operating point
(`tau = 0.88`, 35.4% of traffic to the small tier) the router scores 0.736 and
cuts cost 32% versus always-large. Random routing at the *same* escalation rate
scores **0.7185** (5th–95th percentile over 200 draws: 0.688–0.750).

| | value |
|---|---|
| router quality | 0.7361 |
| random at matched 35.4% rate | 0.7185 |
| difference | +0.0176 |
| draws in which router beats random | **79.5%** |
| share of random→oracle gap captured | **21.9%** |

79.5% is short of the 95% needed to call the difference real, and the router's
quality sits inside random's own 5th–95th interval.

Most of that 32% saving comes from escalating 65% of traffic at all, not from
choosing well which 65%. Reporting the cost reduction without this comparison
would have been misleading, and a policy table showing only always-small,
always-large and the router would have made it look like a success.

**Why it fails.** 88% of items are ones the small model handles, leaving only
58 negatives in 480 — roughly 40 after the train split. TF-IDF over prompt text
cannot learn reasoning difficulty from 40 examples, and reasoning difficulty is
largely not present in a prompt's surface form: two near-identical law questions
differ in whether a 20B model gets them right.

## 3. Two cost findings that do not depend on the router

**The cheap tier can cost more than the expensive one.** With *both* tiers at
default reasoning effort (57-item pilot):

| | mean output tokens | accuracy |
|---|---|---|
| small, 20B | 359.2 | 0.842 |
| large, 120B | 270.2 | 0.895 |

Effective cost ratio **0.97** — routing to the "cheap" model was *more*
expensive. The weaker model compensates for lower capability by reasoning
longer; with output priced 5× input and the tiers only 1.2× apart on the rate
card, that token inflation erased the price advantage entirely.

This is invisible if you reason from the rate card. It is only detectable by
storing realised per-request token counts, which is why `Record` carries them.

**Effort level moves cost more than tier choice does.** Setting the small tier
to `reasoning_effort: low` cut its output from 359 to 98 tokens and restored an
effective cost ratio of **7.62×**. On a reasoning-model pair, "small model" and
"small model configured to be cheap" are different products.

## 4. Latency

p50 improves substantially (2.03s → 0.55s for always-small). **p95 barely moves
— 2.7% at the router's operating point.** The tail is composed of large-model
calls, which routing preserves by design. A "latency router" that reports only
median improvement is telling half the story.

Router overhead, one CPU core:

| | p50 | p95 | p99 |
|---|---|---|---|
| learned, single request | 4.36 ms | 5.63 ms | 7.64 ms |
| learned, batch of 512 | 0.19 ms/item | — | — |
| heuristic | 0.055 ms | 0.085 ms | 0.117 ms |

Charged on every request including escalated ones. Negligible against a 550 ms
model call, but not free — and the heuristic is 79× faster for better ranking.

## 5. What this does and does not show

Shows: the offline-replay methodology works, the cost inversion is real and
mechanistically explained, and prompt-only routing fails on this task with
evidence from three independent measurements.

Does not show: that routing never works. The oracle proves the headroom exists;
the signal is simply not in the prompt. The natural next experiment is a
**cascade** — run the small model first and escalate on its own uncertainty
(logprobs or self-reported confidence) rather than on the input. A model's
output carries far more evidence about whether it struggled than its input does.
`simulate.evaluate(cascade=True)` already charges both tiers and both latencies
for escalated items; the missing piece is capturing a confidence signal during
evaluation.

Also unshown: anything about open-ended generation. Every scorer here requires a
checkable answer, so summarisation, tone, dialogue and creative work — a large
share of real traffic — are untested.

See [FAILURE_MODES.md](FAILURE_MODES.md) for the full list, including a
methodological error where three variables changed between pilots.
