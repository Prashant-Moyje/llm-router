---
title: LLM Cost Latency Router
emoji: 🔀
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# LLM cost/latency router

Routes requests to a cheap or expensive model by predicting whether the cheap
one suffices — and measures that it **doesn't work** on this task.

- **Try the router** — live routing decision, no model call, shown with its real
  reliability
- **Cost/quality frontier** — interactive replay of a 480-item evaluation,
  including the random-at-matched-rate baseline
- **Findings** — the measured results

Built on Groq (`openai/gpt-oss-20b` vs `openai/gpt-oss-120b`).
Source: https://github.com/Prashant-Moyje/llm-router
