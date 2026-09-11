"""Routers: they all answer one question.

    P(the small model is good enough on this request | prompt)

Note what the target is *not*. It is not abstract "task difficulty", which has
no label source and no operational meaning. It is a binary event observed
directly in the offline eval: did the cheap model score at least as well as the
expensive one on this exact item? That reframing is what makes the problem a
standard supervised classification task with an honest label, and it means the
router is automatically re-specified whenever either model tier changes.

Routing rule: send to the small model when p >= tau. tau is chosen on held-out
data by sweeping the cost/quality frontier (see simulate.py), not guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import PromptFeatures, extract


class Router:
    name = "base"

    def predict_proba_small_ok(self, prompts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def route(self, prompts: list[str], tau: float) -> np.ndarray:
        """True == send to the small model."""
        return self.predict_proba_small_ok(prompts) >= tau


class HeuristicRouter(Router):
    """Hand-written baseline. Exists to be beaten.

    Without it there is no way to tell whether the learned model earned its
    keep or whether prompt length alone explains the result.
    """

    name = "heuristic"

    def predict_proba_small_ok(self, prompts: list[str]) -> np.ndarray:
        out = []
        for p in prompts:
            f = dict(zip(_HEURISTIC_KEYS, _pick(extract(p)), strict=True))
            z = (
                2.2
                - 0.55 * f["n_words_log"]
                - 0.80 * f["n_reasoning_cues"]
                + 0.60 * f["n_lookup_cues"]
                - 0.25 * f["n_clauses"]
                - 0.90 * f["has_code_fence"]
                - 0.15 * f["n_math_chars"]
            )
            out.append(1.0 / (1.0 + np.exp(-z)))
        return np.asarray(out)


_HEURISTIC_KEYS = (
    "n_words_log", "n_reasoning_cues", "n_lookup_cues",
    "n_clauses", "has_code_fence", "n_math_chars",
)
_IDX = [1, 10, 11, 9, 7, 6]


def _pick(vec: list[float]) -> list[float]:
    return [vec[i] for i in _IDX]


@dataclass
class LearnedRouter(Router):
    """TF-IDF + handcrafted features -> logistic regression, optionally calibrated.

    Calibration note: thresholding only needs a *monotone* score, so an
    uncalibrated ranker would give the same Pareto frontier. Calibration is
    applied anyway so that tau is interpretable as a probability and can be set
    from a stated risk budget ("escalate anything below 80% confidence") rather
    than read off a plot. That interpretability is the only thing it buys.
    """

    pipeline: Pipeline
    name: str = "learned"

    def predict_proba_small_ok(self, prompts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(prompts))[:, 1]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @staticmethod
    def load(path: Path) -> "LearnedRouter":
        return LearnedRouter(pipeline=joblib.load(path))


def build_pipeline(calibrate: bool = True, seed: int = 13) -> Pipeline:
    text_and_stats = ColumnTransformer(
        transformers=[
            (
                "tfidf_word",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=20_000,
                    sublinear_tf=True, strip_accents="unicode",
                ),
                0,
            ),
            (
                "tfidf_char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 4), min_df=3,
                    max_features=20_000, sublinear_tf=True,
                ),
                0,
            ),
            ("stats", Pipeline([("f", PromptFeatures()), ("s", StandardScaler())]), 0),
        ],
        remainder="drop",
    )
    base = LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=seed
    )
    clf = (
        CalibratedClassifierCV(base, method="sigmoid", cv=3) if calibrate else base
    )
    return Pipeline([("features", _AsColumn()), ("union", text_and_stats), ("clf", clf)])


class _AsColumn(PromptFeatures):
    """Reshape a list of strings into the 2-D object array ColumnTransformer wants."""

    def transform(self, X):  # noqa: N803
        return np.asarray(list(X), dtype=object).reshape(-1, 1)


def save_metadata(path: Path, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
