# One image for the indexer, the API and the Streamlit UI (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    FASTEMBED_CACHE_PATH=/models \
    PATH=/opt/venv/bin:$PATH

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --group ui --no-install-project

COPY src ./src
COPY configs ./configs
COPY prompts ./prompts
COPY data/raw ./data/raw
COPY data/processed ./data/processed
COPY data/golden.jsonl ./data/golden.jsonl
RUN uv sync --frozen --no-dev --group ui

# Bake the embedding model (named in configs/default.yaml) into the image.
RUN python -c "from dpdp_rag.config import load_config; from fastembed import TextEmbedding; \
TextEmbedding(load_config('default.yaml')['embedding']['model'])"

EXPOSE 8000 8501
CMD ["dpdp-api"]
