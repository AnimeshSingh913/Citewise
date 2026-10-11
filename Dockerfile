# syntax=docker/dockerfile:1
FROM python:3.11-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY citewise ./citewise
RUN python -m venv /venv && /venv/bin/pip install ".[s3]"

FROM python:3.11-slim AS runtime
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models FASTEMBED_CACHE_PATH=/models
RUN useradd --create-home --uid 10001 citewise && mkdir -p /data /models && chown citewise /data /models
COPY --from=build /venv /venv
WORKDIR /app
COPY --chown=citewise citewise ./citewise
COPY --chown=citewise frontend ./frontend
COPY --chown=citewise migrations ./migrations
COPY --chown=citewise alembic.ini docker-entrypoint.sh ./
USER citewise
# Bake the embedding models into the image so containers start without a download.
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25')"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
ENTRYPOINT ["sh", "docker-entrypoint.sh"]
CMD ["serve"]
