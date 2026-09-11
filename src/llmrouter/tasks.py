"""Benchmark items and the scorers that grade them.

Design constraint: every scorer here is *programmatically verifiable* -
exact match on a normalised final answer. No LLM-as-judge in the default path.

That is deliberate. The entire project reduces to a difference of two accuracy
numbers, and an LLM judge introduces a noise floor of a few points along with a
known self-preference bias toward whichever model it resembles. With a router
whose whole quality delta may be five points, judge noise would swamp the
signal. The cost of this choice is narrow task coverage - see FAILURE_MODES.md.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ANSWER_INSTRUCTION = (
    "Answer the question. End your reply with a final line of the exact form:\n"
    "FINAL: <answer>"
)


@dataclass
class Task:
    task_id: str
    prompt: str
    reference: str
    source: str
    kind: str  # "numeric" | "choice" | "exact"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

_FINAL_RE = re.compile(r"FINAL:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final(text: str) -> str:
    matches = _FINAL_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    return (text or "").strip().splitlines()[-1].strip() if (text or "").strip() else ""


def _norm_numeric(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").replace("%", "")
    nums = _NUM_RE.findall(s)
    if not nums:
        return None
    try:
        val = float(nums[-1])
    except ValueError:
        return None
    return str(int(val)) if val.is_integer() else f"{val:.6g}"


def score(task: Task, response_text: str) -> float:
    """Return 1.0 for a correct answer, 0.0 otherwise."""
    got = extract_final(response_text)
    if not got:
        return 0.0

    if task.kind == "numeric":
        a, b = _norm_numeric(got), _norm_numeric(task.reference)
        return 1.0 if a is not None and a == b else 0.0

    if task.kind == "choice":
        m = re.search(r"\b([A-Da-d])\b", got)
        return 1.0 if m and m.group(1).upper() == task.reference.strip().upper() else 0.0

    return 1.0 if got.strip().lower() == task.reference.strip().lower() else 0.0


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def load_gsm8k(limit: int = 300, split: str = "test") -> list[Task]:
    from datasets import load_dataset  # imported lazily; heavy dependency

    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.select(range(min(limit, len(ds))))
    out = []
    for i, row in enumerate(ds):
        ref = row["answer"].split("####")[-1].strip()
        out.append(
            Task(
                task_id=f"gsm8k-{i}",
                prompt=f"{row['question']}\n\n{ANSWER_INSTRUCTION}",
                reference=ref,
                source="gsm8k",
                kind="numeric",
            )
        )
    return out


def load_mmlu(limit: int = 300, subjects: tuple[str, ...] = ("all",)) -> list[Task]:
    from datasets import load_dataset

    out: list[Task] = []
    per = max(1, limit // len(subjects))
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test")
        ds = ds.select(range(min(per, len(ds))))
        for i, row in enumerate(ds):
            choices = "\n".join(
                f"{c}. {t}" for c, t in zip("ABCD", row["choices"], strict=False)
            )
            out.append(
                Task(
                    task_id=f"mmlu-{subj}-{i}",
                    prompt=(
                        f"{row['question']}\n{choices}\n\n"
                        "Reply with the single letter of the correct option.\n"
                        f"{ANSWER_INSTRUCTION}"
                    ),
                    reference="ABCD"[row["answer"]],
                    source=f"mmlu/{subj}",
                    kind="choice",
                )
            )
    return out


def load_synthetic(n: int = 600, seed: int = 13) -> list[Task]:
    """Offline stand-in with a genuine easy/hard split.

    Easy: one-step arithmetic. Hard: multi-step word problems.
    For pipeline validation only. Never report metrics from this.
    """
    rng = random.Random(seed)
    tasks: list[Task] = []
    for i in range(n):
        hard = i % 2 == 1
        if hard:
            a, b, c, d = (rng.randint(3, 40) for _ in range(4))
            ans = (a * b + c) * d
            body = (
                f"A warehouse receives {a} pallets holding {b} units each, then "
                f"{c} loose units arrive. If this repeats on each of {d} days, "
                "how many units arrive in total?"
            )
            body = f"[[HARD]] {body}"
        else:
            a, b = rng.randint(2, 60), rng.randint(2, 60)
            ans = a + b
            body = f"What is {a} plus {b}?"
        tasks.append(
            Task(
                task_id=f"syn-{i}",
                prompt=f"{body}\n\n{ANSWER_INSTRUCTION} [[ANSWER=FINAL: {ans}]]",
                reference=str(ans),
                source="synthetic-hard" if hard else "synthetic-easy",
                kind="numeric",
            )
        )
    return tasks


def write_tasks(tasks: list[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(asdict(t)) + "\n")


def read_tasks(path: Path) -> list[Task]:
    with Path(path).open(encoding="utf-8") as fh:
        return [Task(**json.loads(line)) for line in fh if line.strip()]
