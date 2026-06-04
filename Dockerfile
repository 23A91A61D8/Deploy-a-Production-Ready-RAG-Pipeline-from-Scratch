# =============================================================
# RAG Pipeline — Dockerfile
# =============================================================
# Multi-stage build for a lean, production-ready image.

# ---------- Build stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies into a separate directory for easy copy
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- Runtime stage ----------
FROM python:3.11-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/       ./src/
COPY main.py    .
COPY data/      ./data/

# Create directories for vector store persistence
RUN mkdir -p chroma_db faiss_db \
    && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Environment defaults (overridden by docker-compose or .env)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    EMBEDDING_BACKEND=openai \
    VECTOR_STORE_BACKEND=chroma \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    FAISS_PERSIST_DIR=/app/faiss_db \
    DOCS_DIR=/app/data/documents \
    CHUNKING_STRATEGY=sentence_aware \
    CHUNK_SIZE=400 \
    CHUNK_OVERLAP=80 \
    TOP_K=5 \
    LLM_MODEL=gpt-4o-mini

# Expose port (for potential future API/web interface)
EXPOSE 8000

# Default: run ingestion then launch interactive mode
# Override CMD in docker-compose.yml to change behaviour
CMD ["python", "main.py", "--interactive"]
