"""Diagnose provider connectivity before spending a full run.

    python scripts/07_diagnose.py --config configs/groq.yaml

Checks, in order: key present -> endpoint reachable -> key accepted -> the two
configured model IDs actually exist -> a real completion parses and scores.
Each check prints the raw HTTP status and body on failure, which is exactly
what a bulk run cannot show you.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmrouter.config import Config  # noqa: E402
from llmrouter.tasks import ANSWER_INSTRUCTION, Task, score  # noqa: E402

OK, BAD = "  [ok]", "  [FAIL]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/groq.yaml")
    args = ap.parse_args()

    import httpx

    cfg = Config.load(args.config)
    base = cfg.base_url.rstrip("/")
    print(f"config     : {args.config}")
    print(f"base_url   : {base}")
    print(f"key env var: {cfg.api_key_env}")
    print(f"models     : {cfg.small.model_id} | {cfg.large.model_id}\n")

    # 1. key present
    key = os.environ.get(cfg.api_key_env, "")
    print("1. API key in environment")
    if not key:
        print(f"{BAD} {cfg.api_key_env} is empty in THIS shell.")
        print("       $env:GROQ_API_KEY = \"gsk_...\"  (does not persist across windows)")
        return
    print(f"{OK} present, {len(key)} chars, starts {key[:4]!r}")
    if key.strip() != key:
        print(f"{BAD} key has leading/trailing whitespace - a common copy-paste bug")

    client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"})

    # 2 + 3. endpoint reachable and key accepted
    print("\n2. GET /models")
    try:
        r = client.get(f"{base}/models")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} could not reach {base}: {type(exc).__name__}: {exc}")
        return
    print(f"   HTTP {r.status_code}")
    if r.status_code == 401:
        print(f"{BAD} key rejected. Body: {r.text[:300]}")
        print("       Regenerate at https://console.groq.com -> API Keys")
        return
    if r.status_code != 200:
        print(f"{BAD} {r.text[:400]}")
        return

    available = sorted(m["id"] for m in r.json().get("data", []))
    print(f"{OK} key accepted, {len(available)} models visible")

    # 4. configured model IDs exist
    print("\n3. Configured model IDs present on this endpoint")
    missing = []
    for tier, spec in (("small", cfg.small), ("large", cfg.large)):
        if spec.model_id in available:
            print(f"{OK} {tier}: {spec.model_id}")
        else:
            print(f"{BAD} {tier}: {spec.model_id}  NOT FOUND")
            missing.append(spec.model_id)
    if missing:
        print("\n   Models this key can actually use:")
        for m in available:
            print(f"     {m}")
        print("\n   Model IDs get decommissioned. Pick a small and a large one")
        print(f"   from the list above and edit {args.config}.")
        return

    # 5. a real completion, end to end
    print("\n4. Live completion + scoring")
    task = Task("diag", f"What is 6 times 7?\n\n{ANSWER_INSTRUCTION}", "42",
                "diag", "numeric")
    for tier, spec in (("small", cfg.small), ("large", cfg.large)):
        payload = {
            "model": spec.model_id,
            "messages": [{"role": "user", "content": task.prompt}],
            "max_tokens": spec.max_tokens,
        }
        if spec.supports_temperature:
            payload["temperature"] = spec.temperature
        rr = client.post(f"{base}/chat/completions", json=payload)
        if rr.status_code != 200:
            print(f"{BAD} {tier} HTTP {rr.status_code}: {rr.text[:400]}")
            continue
        d = rr.json()
        text = d["choices"][0]["message"].get("content") or ""
        u = d.get("usage", {})
        s = score(task, text)
        print(f"{OK} {tier}: score={s} tokens={u.get('prompt_tokens')}"
              f"/{u.get('completion_tokens')}")
        print(f"       reply tail: {text.strip()[-70:]!r}")
        if s == 0.0:
            print("       scored 0 - model did not emit a parseable 'FINAL:' line.")
            print("       That is a prompt/scorer issue, not a connectivity one.")

    print("\nIf all four passed, the bulk run will work.")


if __name__ == "__main__":
    main()
