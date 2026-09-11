from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmrouter.config import ModelSpec
from llmrouter.features import FEATURE_NAMES, extract
from llmrouter.router import HeuristicRouter, build_pipeline
from llmrouter.simulate import (
    Record,
    always_large,
    always_small,
    evaluate,
    labels,
    oracle,
    random_at_rate,
    retention,
    sweep,
)
from llmrouter.tasks import Task, extract_final, load_synthetic, score

HAIKU = ModelSpec("small", "claude-haiku-4-5", 1.0, 5.0)
OPUS = ModelSpec("large", "claude-opus-5", 5.0, 25.0, supports_temperature=False)


def _rec(i, s, l, sc=0.001, lc=0.005, sl=0.3, ll=1.5):
    return Record(f"t{i}", f"prompt {i}", "syn", s, l, sc, lc, sl, ll)


# --------------------------------------------------------------------- scoring


def test_numeric_scoring_normalises_formatting():
    t = Task("a", "q", "1200", "syn", "numeric")
    assert score(t, "FINAL: 1,200") == 1.0
    assert score(t, "FINAL: $1200.00") == 1.0
    assert score(t, "FINAL: 1201") == 0.0


def test_choice_scoring_is_letter_exact():
    t = Task("a", "q", "C", "syn", "choice")
    assert score(t, "reasoning...\nFINAL: C") == 1.0
    assert score(t, "FINAL: B") == 0.0


def test_extract_final_takes_last_marker():
    assert extract_final("FINAL: 1\nmore\nFINAL: 2") == "2"


def test_empty_response_scores_zero_not_crash():
    assert score(Task("a", "q", "5", "syn", "numeric"), "") == 0.0


# -------------------------------------------------------------------- pricing


def test_cost_uses_reported_tokens():
    # 1M in, 1M out on Haiku 4.5 -> $1 + $5
    assert HAIKU.cost_usd(1_000_000, 1_000_000) == pytest.approx(6.0)
    assert OPUS.cost_usd(1_000_000, 1_000_000) == pytest.approx(30.0)


# ------------------------------------------------------------------- features


def test_feature_vector_length_matches_names():
    assert len(extract("hello world 42")) == len(FEATURE_NAMES)


def test_features_are_prompt_only():
    """Regression guard against answer leakage into the feature path."""
    a = extract("Explain step by step why 17 is prime.")
    b = extract("Explain step by step why 17 is prime.")
    assert a == b  # deterministic, no hidden state


# ---------------------------------------------------------------- policy math


def test_always_large_beats_always_small_when_it_should():
    recs = [_rec(i, 0.0, 1.0) for i in range(10)]
    assert evaluate(recs, always_small(recs), "s").quality == 0.0
    assert evaluate(recs, always_large(recs), "l").quality == 1.0


def test_oracle_is_an_upper_bound_on_quality():
    rng = np.random.default_rng(0)
    recs = [
        _rec(i, float(rng.integers(0, 2)), float(rng.integers(0, 2)))
        for i in range(200)
    ]
    q_oracle = evaluate(recs, oracle(recs), "o").quality
    for policy in (always_small(recs), always_large(recs), random_at_rate(recs, 0.5)):
        assert evaluate(recs, policy, "p").quality <= q_oracle + 1e-9


def test_cascade_charges_both_tiers_on_escalation():
    recs = [_rec(0, 0.0, 1.0, sc=0.001, lc=0.005)]
    to_small = np.array([False])
    direct = evaluate(recs, to_small, "direct", cascade=False, n_boot=10)
    casc = evaluate(recs, to_small, "cascade", cascade=True, n_boot=10)
    assert direct.cost_per_1k_usd == pytest.approx(5.0)
    assert casc.cost_per_1k_usd == pytest.approx(6.0)
    assert casc.latency_p50_s == pytest.approx(1.8)


def test_labels_flag_small_sufficient():
    recs = [_rec(0, 1.0, 1.0), _rec(1, 0.0, 1.0), _rec(2, 1.0, 0.0)]
    assert labels(recs).tolist() == [1, 0, 1]


def test_retention_is_nan_when_tiers_tie():
    assert np.isnan(retention(0.5, 0.5, 0.5))


def test_sweep_is_monotone_in_cost():
    """Raising tau routes strictly less traffic to the cheap tier."""
    recs = [_rec(i, 1.0, 1.0) for i in range(50)]
    probs = np.linspace(0, 1, 50)
    rows = sweep(recs, probs, taus=np.linspace(0, 1, 11))
    shares = [r["small_share"] for r in rows]
    assert shares == sorted(shares, reverse=True)


# --------------------------------------------------------------------- models


def test_heuristic_prefers_small_for_lookups():
    h = HeuristicRouter()
    easy = "What is the capital of France?"
    hard = (
        "Derive step by step why this refactor changes complexity, analyze the "
        "edge case, and justify the trade-off:\n```python\ndef f(): pass\n```"
    )
    p = h.predict_proba_small_ok([easy, hard])
    assert p[0] > p[1]


def test_learned_pipeline_trains_and_ranks():
    tasks = load_synthetic(n=160, seed=1)
    X = [t.prompt for t in tasks]
    y = np.array([0 if "[[HARD]]" in p else 1 for p in X])
    pipe = build_pipeline(calibrate=False, seed=1)
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert p[y == 1].mean() > p[y == 0].mean()


def test_probabilities_in_unit_interval():
    tasks = load_synthetic(n=60, seed=2)
    p = HeuristicRouter().predict_proba_small_ok([t.prompt for t in tasks])
    assert p.min() >= 0.0 and p.max() <= 1.0


# ----------------------------------------------------- OpenAI-compat provider


def _stub_server(handler_calls, throttle_first=True):
    """Local OpenAI-compatible stub. Returns (server, thread)."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: ANN002
            pass

        def do_POST(self):  # noqa: N802
            handler_calls["n"] += 1
            self.rfile.read(int(self.headers["Content-Length"]))
            if throttle_first and handler_calls["n"] == 1:
                self.send_response(429)
                self.send_header("retry-after", "1")
                self.end_headers()
                return
            out = {
                "choices": [{"message": {"content": "FINAL: 42"}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 8},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_openai_compat_retries_429_and_bills_reported_tokens():
    from llmrouter.providers import OpenAICompatProvider

    calls = {"n": 0}
    srv = _stub_server(calls)
    try:
        port = srv.server_address[1]
        p = OpenAICompatProvider(f"http://127.0.0.1:{port}/v1", require_key=False)
        spec = ModelSpec("s", "llama-3.1-8b-instant", 0.05, 0.08,
                         pricing_basis="imputed")
        g = p.generate("What is 6 times 7?", spec)
    finally:
        srv.shutdown()

    assert g.error is None
    assert g.text == "FINAL: 42"
    assert calls["n"] == 2  # throttled once, then succeeded
    assert g.cost_usd == pytest.approx((120 * 0.05 + 8 * 0.08) / 1e6)


def test_latency_excludes_retry_backoff():
    """Regression guard.

    A 429 with Retry-After: 1 must not be charged to the model's latency. If
    the timer starts before the retry loop, this records ~1s for a request the
    stub answers instantly, and every latency number on a rate-limited free
    tier becomes a measurement of throttling instead of inference.
    """
    from llmrouter.providers import OpenAICompatProvider

    calls = {"n": 0}
    srv = _stub_server(calls)
    try:
        port = srv.server_address[1]
        p = OpenAICompatProvider(f"http://127.0.0.1:{port}/v1", require_key=False)
        g = p.generate("x", ModelSpec("s", "m", 0.05, 0.08))
    finally:
        srv.shutdown()

    assert calls["n"] == 2
    assert g.latency_s < 0.25, f"retry sleep leaked into latency: {g.latency_s}s"


def test_missing_api_key_fails_loudly():
    from llmrouter.providers import OpenAICompatProvider

    with pytest.raises(RuntimeError, match="NOPE_KEY"):
        OpenAICompatProvider("http://localhost:1/v1", api_key_env="NOPE_KEY")


# ------------------------------------------------------------------ repricing


def test_reprice_recomputes_from_token_counts():
    from llmrouter.simulate import reprice

    recs = [Record("t", "p", "s", 1.0, 1.0, 0.0, 0.0, 0.3, 1.5,
                   small_input_tokens=1_000_000, small_output_tokens=0,
                   large_input_tokens=1_000_000, large_output_tokens=0)]
    out = reprice(recs, HAIKU, OPUS)
    assert out[0].small_cost_usd == pytest.approx(1.0)
    assert out[0].large_cost_usd == pytest.approx(5.0)


def test_reprice_preserves_measured_quality_and_latency():
    """Repricing is a cost counterfactual only. It must never move quality."""
    from llmrouter.simulate import reprice

    recs = [Record("t", "p", "s", 0.0, 1.0, 9.9, 9.9, 0.31, 1.52,
                   small_input_tokens=100, small_output_tokens=20,
                   large_input_tokens=100, large_output_tokens=20)]
    out = reprice(recs, HAIKU, OPUS)
    assert out[0].small_score == 0.0 and out[0].large_score == 1.0
    assert out[0].small_latency_s == 0.31 and out[0].large_latency_s == 1.52


def test_reprice_refuses_records_without_token_counts():
    from llmrouter.simulate import reprice

    recs = [Record("t", "p", "s", 1.0, 1.0, 0.001, 0.005, 0.3, 1.5)]
    with pytest.raises(ValueError, match="older version"):
        reprice(recs, HAIKU, OPUS)


def test_extra_params_reach_the_request_payload():
    """reasoning_effort must actually be sent, not silently dropped.

    A config knob that never reaches the wire looks like it works: the run
    completes, numbers change slightly from noise, and the cost inversion it
    was meant to fix stays unfixed.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: ANN002
            pass

        def do_POST(self):  # noqa: N802
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            out = {"choices": [{"message": {"content": "FINAL: 1"}}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())

    from llmrouter.providers import OpenAICompatProvider

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        p = OpenAICompatProvider(f"http://127.0.0.1:{srv.server_address[1]}/v1",
                                 require_key=False)
        spec = ModelSpec("s", "openai/gpt-oss-20b", 0.1, 0.5,
                         extra_params={"reasoning_effort": "low"})
        p.generate("x", spec)
    finally:
        srv.shutdown()

    assert seen.get("reasoning_effort") == "low"
    assert seen.get("model") == "openai/gpt-oss-20b"


# --------------------------------------------------------------- paired test


def test_mcnemar_flat_when_discordant_pairs_are_balanced():
    """4 vs 5 discordant pairs must not read as a quality difference."""
    from llmrouter.simulate import paired_test

    recs = (
        [_rec(i, 1.0, 1.0) for i in range(41)]
        + [_rec(100 + i, 1.0, 0.0) for i in range(4)]
        + [_rec(200 + i, 0.0, 1.0) for i in range(5)]
        + [_rec(300 + i, 0.0, 0.0) for i in range(10)]
    )
    out = paired_test(recs)
    assert out["discordant_pairs"] == 9
    assert out["mcnemar_exact_p"] > 0.9
    assert out["significant_at_05"] is False


def test_mcnemar_detects_a_real_one_sided_gap():
    from llmrouter.simulate import paired_test

    recs = (
        [_rec(i, 1.0, 1.0) for i in range(40)]
        + [_rec(100 + i, 0.0, 1.0) for i in range(20)]
        + [_rec(200 + i, 1.0, 0.0) for i in range(2)]
    )
    out = paired_test(recs)
    assert out["significant_at_05"] is True


def test_concordant_items_carry_no_signal():
    """Adding items both tiers get right must not change the p-value."""
    from llmrouter.simulate import paired_test

    base = [_rec(100 + i, 1.0, 0.0) for i in range(3)] + [
        _rec(200 + i, 0.0, 1.0) for i in range(3)
    ]
    padded = base + [_rec(900 + i, 1.0, 1.0) for i in range(500)]
    assert paired_test(base)["mcnemar_exact_p"] == paired_test(padded)[
        "mcnemar_exact_p"
    ]
