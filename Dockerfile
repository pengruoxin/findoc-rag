# Web image: starts with the committed demo index and includes the local OCR
# runtime so uploaded native and scanned PDFs can enter the same evidence path.
# The optional dense stack remains excluded; lexical retrieval is the default.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Resolve dependencies before copying sources so edits do not invalidate the layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra api --extra ocr

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs
COPY data/evaluation ./data/evaluation
RUN uv sync --frozen --extra api --extra ocr

# Build the demo index at image-build time so the container starts ready.
# Verifies the external SHA lock first; a tampered evidence file fails the build.
RUN python scripts/build_demo_index.py --index-root /app/data/indexes/demo

# Run unprivileged. data/ needs to be writable for traces and upload jobs.
RUN useradd --create-home --uid 10001 findoc \
    && mkdir -p /app/data/traces /app/data/cache /app/data/uploads \
    && chown -R findoc:findoc /app/data
USER findoc

ENV FINDOC_RAG_HOST=0.0.0.0 \
    FINDOC_RAG_PORT=8000 \
    FINDOC_RAG_INDEX_DIR=/app/data/indexes/demo \
    FINDOC_RAG_DEFAULT_MODE=lexical

EXPOSE 8000

# The API has no built-in user authentication. Keep it behind an authenticated
# reverse proxy when it is reachable outside a trusted network.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready',timeout=2).status==200 else 1)"

CMD ["findoc-rag", "serve"]
