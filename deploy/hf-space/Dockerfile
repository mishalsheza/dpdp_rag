# Hugging Face Space image: builds the app from the GitHub repo and runs the API and the
# Streamlit UI in one container. See docs/DEPLOY.md.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    FASTEMBED_CACHE_PATH=/models \
    DPDP_CONFIG_OVERRIDES=hf.yaml \
    HOME=/tmp \
    PATH=/opt/venv/bin:$PATH

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

ARG REPO=https://github.com/mishalsheza/dpdp_rag
ARG REF=main
RUN git clone --depth 1 --branch "$REF" "$REPO" /app
WORKDIR /app
RUN uv sync --frozen --no-dev --group ui

# Bake the embedding model (named in configs/default.yaml) into the image.
RUN python -c "from dpdp_rag.config import load_config; from fastembed import TextEmbedding; \
TextEmbedding(load_config('default.yaml')['embedding']['model'])"

COPY start.sh /start.sh
# Spaces run the container as uid 1000: let it write metrics, caches and the model cache.
RUN chmod +x /start.sh && mkdir -p /app/data/cache && chmod -R a+rwX /app/data /models

EXPOSE 7860
CMD ["/start.sh"]
