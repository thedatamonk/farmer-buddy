# ---- Build stage ----
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --frozen --no-dev

COPY data/ data/
COPY scripts/ scripts/

# ---- Runtime stage ----
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY --from=builder /app/src src
COPY --from=builder /app/data data
COPY --from=builder /app/scripts scripts

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

CMD ["uvicorn", "kisan.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
