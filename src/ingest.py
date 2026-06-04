"""
ingest.py
---------
Ingestion Phase: load → chunk → embed → store.

Can be run as a standalone script:
  python -m src.ingest --docs_dir ./data/documents

Or called programmatically via ingest_documents().
"""

import os
import logging
import argparse
from typing import List, Dict

from src.document_loader import load_documents_from_directory
from src.chunker import chunk_documents
from src.embedder import Embedder
from src.vector_store import get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_documents(
    docs_dir: str,
    chunking_strategy: str = "sentence_aware",
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    embedding_backend: str | None = None,
    vector_store_backend: str | None = None,
    batch_size: int = 64,
) -> Dict[str, int]:
    """
    Full ingestion pipeline.

    Parameters
    ----------
    docs_dir            : Directory containing source documents
    chunking_strategy   : "fixed_size" | "sentence_aware" | "sliding_window"
    chunk_size          : Target chunk size in tokens
    chunk_overlap       : Overlap tokens between consecutive chunks
    embedding_backend   : "openai" | "huggingface" (None → env var)
    vector_store_backend: "chroma" | "faiss"       (None → env var)
    batch_size          : Number of chunks to embed per API call

    Returns
    -------
    dict with "documents_loaded", "chunks_created", "chunks_stored"
    """

    # ------------------------------------------------------------------
    # 1. Load documents
    # ------------------------------------------------------------------
    logger.info("=== STEP 1: Loading documents from '%s' ===", docs_dir)
    documents = load_documents_from_directory(docs_dir)
    if not documents:
        logger.error("No documents found in '%s'. Aborting.", docs_dir)
        return {"documents_loaded": 0, "chunks_created": 0, "chunks_stored": 0}

    # ------------------------------------------------------------------
    # 2. Chunk documents
    # ------------------------------------------------------------------
    logger.info(
        "=== STEP 2: Chunking (%s, size=%d, overlap=%d) ===",
        chunking_strategy, chunk_size, chunk_overlap,
    )
    chunks = chunk_documents(
        documents,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info("Created %d chunks from %d documents.", len(chunks), len(documents))

    # ------------------------------------------------------------------
    # 3. Embed chunks
    # ------------------------------------------------------------------
    logger.info("=== STEP 3: Embedding chunks ===")
    embedder = Embedder(backend=embedding_backend)
    all_embeddings: List[List[float]] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        logger.info(
            "  Embedding batch %d/%d (%d chunks)…",
            i // batch_size + 1,
            (len(chunks) - 1) // batch_size + 1,
            len(batch),
        )
        vecs = embedder.embed(texts)
        all_embeddings.extend(vecs)

    logger.info("Embedded %d chunks.", len(all_embeddings))

    # ------------------------------------------------------------------
    # 4. Store in vector DB
    # ------------------------------------------------------------------
    logger.info("=== STEP 4: Storing in vector store ===")
    store = get_vector_store(backend=vector_store_backend)
    store.add_chunks(chunks, all_embeddings)
    logger.info("Stored %d chunks. Total in store: %d", len(chunks), store.count())

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored": len(chunks),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=os.getenv("DOCS_DIR", "./data/documents"),
        help="Path to the directory containing source documents.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=os.getenv("CHUNKING_STRATEGY", "sentence_aware"),
        choices=["fixed_size", "sentence_aware", "sliding_window"],
        help="Chunking strategy to use.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", 400)),
        help="Target chunk size in tokens.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", 80)),
        help="Overlap tokens between chunks.",
    )
    args = parser.parse_args()

    stats = ingest_documents(
        docs_dir=args.docs_dir,
        chunking_strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print("\n✅ Ingestion complete:")
    print(f"   Documents loaded : {stats['documents_loaded']}")
    print(f"   Chunks created   : {stats['chunks_created']}")
    print(f"   Chunks stored    : {stats['chunks_stored']}")
