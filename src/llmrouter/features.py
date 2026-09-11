"""Prompt-only features for the router.

Hard constraint: the router sees the prompt and nothing else. At serve time
there is no reference answer and no model output yet, so any feature computed
from either would leak and inflate offline metrics. Everything here is
computable in microseconds from the raw request string.

Router latency is part of the product, not an implementation detail. If the
router costs 300 ms, it cancels most of the latency win it exists to create.

Measured on one CPU core (see scripts/06_bench_latency.py):

    TF-IDF + LR, one request at a time   p50 2.97 ms   p95 3.18 ms
    TF-IDF + LR, batch of 512            0.09 ms/item
    handcrafted features only            0.03 ms

Roughly 3 ms of the single-request path is scikit-learn vectorizer overhead on
a batch of one, not feature computation. That is acceptable against a
350 ms-and-up model call, but it is not free and it is not sub-millisecond. A
sentence-embedding router would add tens of milliseconds and a model load; that
is the trade this feature set is buying out of.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_REASONING_CUES = (
    "step by step", "prove", "derive", "explain why", "compare", "trade-off",
    "tradeoff", "analyze", "analyse", "justify", "critique", "optimize",
    "debug", "refactor", "edge case", "why does", "how would",
)
_LOOKUP_CUES = (
    "what is", "who is", "when did", "define", "list", "translate",
    "capital of", "summarize in one", "yes or no", "true or false",
)
_MATH_CHARS = set("+-*/^=<>%∑∫√")

FEATURE_NAMES = [
    "n_chars_log", "n_words_log", "n_sentences", "mean_word_len",
    "n_digits_log", "digit_ratio", "n_math_chars", "has_code_fence",
    "n_questions", "n_clauses", "n_reasoning_cues", "n_lookup_cues",
    "n_numbers", "max_number_log", "uppercase_ratio", "n_newlines",
]


def extract(text: str) -> list[float]:
    t = text or ""
    words = t.split()
    n_words = len(words)
    digits = sum(c.isdigit() for c in t)
    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)] or [0.0]
    lower = t.lower()
    return [
        np.log1p(len(t)),
        np.log1p(n_words),
        float(len(re.findall(r"[.!?]", t))),
        float(np.mean([len(w) for w in words])) if n_words else 0.0,
        np.log1p(digits),
        digits / max(1, len(t)),
        float(sum(c in _MATH_CHARS for c in t)),
        1.0 if "```" in t else 0.0,
        float(t.count("?")),
        float(len(re.findall(r"[,;:]", t))),
        float(sum(c in lower for c in _REASONING_CUES)),
        float(sum(c in lower for c in _LOOKUP_CUES)),
        float(len(numbers)),
        float(np.log1p(abs(max(numbers, key=abs)))),
        sum(c.isupper() for c in t) / max(1, len(t)),
        float(t.count("\n")),
    ]


class PromptFeatures(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper so features live inside the fitted pipeline.

    Keeping feature code inside the pickled pipeline means the serving process
    cannot drift from the training process - a top source of silent train/serve
    skew when feature logic is duplicated in an API handler.
    """

    def fit(self, X, y=None):  # noqa: N803, ARG002
        return self

    def transform(self, X):  # noqa: N803
        return np.asarray([extract(x) for x in X], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002
        return np.asarray(FEATURE_NAMES, dtype=object)
