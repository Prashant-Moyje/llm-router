FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Serving needs a much smaller dependency set than training. datasets/
# matplotlib/pytest are build-time only and would roughly triple the image.
COPY requirements.txt .
RUN grep -vE '^(datasets|matplotlib|pytest|httpx)' requirements.txt > serve-reqs.txt \
    && pip install --no-cache-dir -r serve-reqs.txt

COPY src/ ./src/
COPY configs/ ./configs/
# The trained router is baked in so the container is self-contained and the
# deployed artifact is pinned to a known evaluation run.
COPY artifacts/router.joblib ./artifacts/router.joblib
COPY artifacts/router_meta.json ./artifacts/router_meta.json

ENV ROUTER_CONFIG=configs/default.yaml \
    ROUTER_MODEL=artifacts/router.joblib \
    ROUTER_PROVIDER=anthropic \
    ROUTER_TAU=0.50

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

RUN useradd -m -u 10001 appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "llmrouter.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
