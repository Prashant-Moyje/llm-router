# Failure modes and null results

Written to be read by someone trying to find the holes. Where a limit is
measured, the number is here. Where it is only reasoned about, that is stated.

---

## 1. Null result: the router does not generalise across task sources

**This is the most important finding in the repo and it is negative.**

`scripts/03_train_router.py` reports two splits. Stratified random split:

| split | AUC | AP | Brier |
|---|---|---|---|
| learned | 0.856 | 0.917 | 0.126 |
| heuristic baseline | 0.852 | 0.882 | 0.260 |

Grouped split, holding out an entire task source the router never saw:

| split | AUC | AP | Brier |
|---|---|---|---|
| learned, unseen source | **0.449** | 0.291 | 0.692 |

AUC below 0.5 on unseen sources — **worse than a coin flip**.

The interpretation: on the random split the router is largely learning the
surface form that separates one benchmark from another (`gsm8k` prompts look
like `gsm8k` prompts), and difficulty correlates with source. Strip that cue
away and there is little within-source discrimination left. The stratified 0.856
is real but it is measuring something narrower than "predicts difficulty".

Consequence: **do not deploy this against a traffic mix unlike the training
mix.** The honest claim is "learns the small model's competence boundary on the
task distribution it was trained on", not "predicts task difficulty".

Note also that the learned router beats the six-line heuristic by 0.004 AUC on
the stratified split. On this data, TF-IDF plus logistic regression is barely
earning its complexity over a hand-written rule. It wins clearly on Brier
(0.126 vs 0.260), so it is much better *calibrated*, which is what makes τ
interpretable — but on ranking alone, the heuristic is nearly as good and 85×
faster. If you only need ranking, ship the heuristic.

---

## 2. Verifiable-only scoring buys reliability and pays in coverage

Every scorer is exact match on a normalised final answer. No LLM judge.

Why: the entire result is a difference of two accuracy numbers, sometimes only
a few points wide. An LLM judge adds a noise floor of several points plus known
self-preference bias toward outputs resembling its own, which would swamp the
signal.

The cost: only tasks with a checkable answer are covered — arithmetic,
multiple choice, extraction. Open-ended generation, summarisation, tone,
creative writing, and multi-turn dialogue are **not measured at all**, and
those are a large share of real LLM traffic. Savings measured here do not
transfer to that traffic without re-measurement.

## 3. The savings headline is mostly a function of the price ratio

Haiku 4.5 vs Opus 5 is a 5× price gap; Haiku vs Sonnet 5 is 2×. The same router
with identical routing decisions produces a dramatically smaller headline under
`configs/sonnet_large.yaml`. Run both before quoting a number. "Cut cost 40%"
without naming the tier pair is not a claim about the router.

## 4. p95 latency is not improved

Median latency drops because most requests go to the fast tier. The tail is
composed of large-model calls, which routing preserves by design. Reported
directly in the results table; do not let a "latency router" title imply
otherwise.

The router also adds ~3 ms p50 to every request including escalated ones. On a
cascade this compounds: escalated requests pay the router, the small model, and
the large model.

## 5. Cascade accounting is where these systems get oversold

`--cascade` runs the small model first and escalates on low confidence. Escalated
items pay **both** bills and **both** latencies. `evaluate(cascade=True)`
charges this and `test_cascade_charges_both_tiers_on_escalation` locks it in.
Cascade only wins when the escalation rate is low enough that the doubled cost
on escalated items is outweighed by the savings on the rest. It is not
free accuracy.

## 6. Failed API calls are dropped, not scored zero

`02_run_offline_eval.py` records errors and excludes those items. Scoring a
timeout as an incorrect answer would attribute an infrastructure fault to model
quality and bias the labels toward whichever tier is rate-limited harder. The
count is reported as `dropped_errors`. If it is more than a percent or two, the
run is not trustworthy — investigate before using the records.

## 7. Cross-model token counts are not comparable

Claude tiers do not share a tokenizer; Sonnet 5's produces roughly 30% more
tokens for the same text than Haiku 4.5's. Cost is therefore always computed
from the token counts the **API reports**, never from a local tokenizer. Any
reimplementation that estimates tokens locally will mis-bill at least one tier
and distort the entire comparison.

## 8. Single-sample scoring, temperature 0

Each item is evaluated once. Run-to-run variance is unmeasured, so the accuracy
figures carry sampling noise beyond the bootstrap CIs, which only capture
*item* sampling and not *generation* sampling. Multi-sample evaluation would fix
this and multiply the eval cost by the sample count.

Also: `supports_temperature` is per-model because Opus 5 and Sonnet 5 reject
`temperature` and `top_p` outright (a 400, not a warning). Determinism is
therefore not fully controllable on those tiers.

## 9. τ is selected on the test split

`04_simulate.py` sweeps τ on the held-out split and picks the cheapest point
clearing a quality floor. That point is therefore mildly optimistic — the
threshold saw the data it is reported on. A three-way train/validation/test
split would fix it at the cost of statistical power on a few-hundred-item
benchmark. With n=180 the choice was power; with a few thousand items, split
three ways.

## 10. The label is a tie-break, not a quality measure

`small_ok = (small_score >= large_score)`. On binary scoring both models are
often *both wrong* (0 vs 0), which labels as "small sufficient". That is
correct for cost purposes — if both fail, pay less to fail — but it means the
positive class mixes "small succeeded" with "neither worked". On a
low-accuracy benchmark this inflates the positive rate and the router partly
learns to spot items *nobody* can do.

## 11. No safety or PII routing

Purely cost/quality. It does not consider that some requests should go to the
stronger model regardless of predicted sufficiency — anything user-facing and
high-stakes, anything legally sensitive. Production use needs an override list
that bypasses the router entirely, ahead of the probability check.

## 12. Free-tier dollar figures are imputed, not billed

`configs/groq.yaml` runs on Groq's free tier, where inference costs \$0. The
dollar figures are the provider's **published pay-as-you-go rates applied to
measured token counts** - a model of what the traffic would cost, not a bill.
`pricing_basis: imputed` records this and is echoed into
`reports/offline_eval.json` so a figure can never be mistaken for spend.

`configs/ollama.yaml` is worse in this respect: `pricing_basis: local` means
the per-token prices are borrowed from comparable hosted models purely so the
cost axis has units. **Cost savings from a local run are a simulation.**
Latency from a local run, by contrast, is genuinely measured and is the more
defensible metric to report from that config.

## 13. Free-tier latency is not production latency

Free tiers run on shared queues with aggressive rate limits (Groq: ~30 req/min).
Measured latency there reflects queueing and throttling as much as inference.
The provider excludes retry backoff from the reported number - see
`test_latency_excludes_retry_backoff`, which exists because the first
implementation got this wrong and charged a 1-second `Retry-After` sleep to
model latency. But queue delay on a shared free tier is still inside the
measurement and cannot be separated out.

Use `--workers 2` on a free tier. High concurrency there produces more
throttling, not more throughput, and inflates every latency number.

## 15. Counterfactual repricing is a sensitivity check, not a result

`04_simulate.py --price-config X` recomputes cost from stored token counts at
another provider's rates. It exists because a provider's price ladder may not
suit the demonstration: Groq's remaining chat tiers differ by ~1.4x blended, so
a cost headline measured there is capped near 30% regardless of how good the
router is.

Two hard limits on its use:

1. **Token counts are not transferable across model families.** Different
   tokenizers produce different counts for identical text. Repricing gpt-oss
   token counts at Claude rates inherits that error, on top of the fact that
   the two families would not have produced the same outputs at all.
2. **It cannot be used to quote a bigger number.** The legitimate use is
   showing that the savings figure moves with the price ratio while the routing
   decisions stay fixed - i.e. demonstrating §3. Reporting a repriced figure as
   the project's result would be dishonest.

`headline.json` records `pricing` as either `measured (...)` or
`COUNTERFACTUAL: ...` so the two can never be confused after the fact.

The defensible split when running on a narrow-price-gap provider: report
**quality and latency as measured** (those are real), and present cost as a
sensitivity curve across price ratios rather than a single headline number.

## 17. Measured null result: the cheap tier cost MORE than the expensive one

57-item MMLU pilot, Groq, gpt-oss-20b vs gpt-oss-120b, both at **default
reasoning effort**:

| | mean output tokens | accuracy |
|---|---|---|
| small, 20B | 359.2 | 0.842 |
| large, 120B | 270.2 | 0.895 |

`effective_cost_ratio: 0.97` - routing to the small tier was **more
expensive**, not cheaper.

Mechanism: the weaker model compensates for lower capability by reasoning
longer. Output is priced 5x input, and these two tiers sit only 1.2x apart on
the rate card, so a 33% output-token increase more than erases the price
advantage. **The premise of cost routing inverts.**

This generalises beyond Groq. Any small/large pair where (a) both tiers are
reasoning models, (b) output is priced well above input, and (c) the tier price
ratio is compressed, is a candidate for the same inversion. It is invisible if
you reason from the rate card instead of measuring realised token counts, which
is the entire argument for storing per-record token counts (§15).

The control is `reasoning_effort`, exposed as `extra_params` per tier.
`configs/groq.yaml` sets the small tier to `low`; `configs/groq_default_effort.yaml`
preserves the configuration that produced the inversion so it stays
reproducible. **Report both.** The corrected run is the engineering result; the
inverted run is the finding.

## 18. Subject saturation is relative to the model pair

The same pilot: `abstract_algebra`, `college_physics` and `formal_logic` all
returned label_rate 1.0 - both tiers perfect, zero signal. `professional_medicine`
returned small_acc 1.0 vs large_acc 0.889, i.e. the small model **beat** the
large one, which at n=9 is well inside noise.

Consequences: a benchmark's difficulty is a property of the model pair, not of
the benchmark, so `MMLU_SUBJECTS` cannot be treated as portable. And with 9-10
items per subject those label rates carry roughly +/-0.15 of sampling error -
enough to reverse a subject's apparent ranking. Prune on a pilot, but pilot
larger than 10 per subject before trusting the ordering.

## 19. Methodological error: three variables changed at once

Between the two pilots, the small tier's reasoning effort, the large tier's
reasoning effort, AND the subject list all changed. The runs are therefore not
comparable and no observed difference can be attributed to any single cause.
This was a design error, recorded rather than quietly corrected.

`configs/groq_small_low.yaml` isolates the small-tier lever with the large tier
at default. The intended sequence, on an identical task file, is A (both
default) -> B (small=low only) -> C (both changed).

Note also that `02_run_offline_eval.py` resumes by `task_id` and will merge
records produced under different configs without complaint, silently mixing
conditions. Copy `records.jsonl` to a distinct filename after each run.

## 20. Measured null result: no detectable quality gap between the tiers

60-item MMLU pilot, gpt-oss-20b at `low` effort vs gpt-oss-120b at `high`:

| | accuracy | mean output tokens |
|---|---|---|
| small | 0.750 | 92.8 |
| large | 0.767 | 862.0 |

`effective_cost_ratio: 8.05` - the large tier cost 8x more.

The accuracy difference does not survive a paired test. The tiers are scored on
identical items, so the outcomes are paired and only discordant pairs carry
information:

    both correct       41
    both wrong         10
    small only correct  4
    large only correct  5
    ------------------------------
    discordant pairs    9      McNemar exact p = 1.00

The "1.7 point gap" is five items versus four. **8x the cost for no measurable
quality benefit.**

If this holds at full n, the correct policy for this pair is always-small, and
a router adds nothing: routing has value only in proportion to the quality gap
it preserves, and here there is no gap to preserve. That is a legitimate
finding about *this model pair on this benchmark* and should be reported as
such - not smoothed over by picking a threshold that appears to trade cost for
quality when the quality axis is flat.

Note what this does NOT license: concluding that the 120B is never worth its
price. MMLU may simply be saturated relative to both tiers. A benchmark where
the larger model actually separates (harder reasoning, long-horizon coding)
could show a very different picture, and testing that is the obvious next step.

## 21. Mock provider numbers are not results

`MockProvider` fabricates a capability gap so the pipeline can be validated
offline for free. `02_run_offline_eval.py` prints a warning when it is used.
Any number in the README traceable to `"provider": "mock"` in
`reports/offline_eval.json` is a placeholder, not a finding.
