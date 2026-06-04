"""
vector_store.py
---------------
Persists chunk embeddings and supports similarity search.

Supports two backends (controlled by VECTOR_STORE_BACKEND env var):
  - "chroma"  : ChromaDB  (default, persistent)
  - "faiss"   : FAISS     (file-based persistence)
"""

import os
import json
import pickle
import logging
import uuid
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity (fallback when numpy not available)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------

class ChromaVectorStore:
    """
    Persistent vector store backed by ChromaDB.
    Data is written to *persist_directory*.
    """

    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "./chroma_db",
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            ) from exc

        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore ready — collection='%s', persist_dir='%s'",
            collection_name,
            persist_directory,
        )

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        """Store chunks with their pre-computed embeddings."""
        if not chunks:
            return

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c.get("source", "unknown"),
                "chunk_index": int(c.get("chunk_index", 0)),
                "strategy": c.get("strategy", "unknown"),
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("ChromaVectorStore: added %d chunks.", len(chunks))

    def similarity_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict]:
        """Return top-K most similar chunks with their metadata."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": round(1 - dist, 4),  # convert distance → similarity
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self.collection.delete(where={"chunk_index": {"$gte": 0}})
        logger.warning("ChromaVectorStore: collection cleared.")


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------

class FAISSVectorStore:
    """
    Vector store backed by FAISS with JSON-file metadata storage.
    Persists index + metadata to *persist_directory*.
    """

    INDEX_FILE = "faiss_index.bin"
    META_FILE = "faiss_metadata.pkl"

    def __init__(self, persist_directory: str = "./faiss_db", dimension: int = 1536):
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu and numpy not installed. "
                "Run: pip install faiss-cpu numpy"
            ) from exc

        self._faiss = faiss
        self._np = np
        self.persist_directory = persist_directory
        self.dimension = dimension
        os.makedirs(persist_directory, exist_ok=True)

        index_path = os.path.join(persist_directory, self.INDEX_FILE)
        meta_path = os.path.join(persist_directory, self.META_FILE)

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata: List[Dict] = pickle.load(f)
            logger.info(
                "FAISSVectorStore: loaded existing index (%d vectors).", len(self.metadata)
            )
        else:
            # Inner-product index (use normalised vectors → cosine similarity)
            self.index = faiss.IndexFlatIP(dimension)
            self.metadata = []
            logger.info("FAISSVectorStore: created new index (dim=%d).", dimension)

    def _persist(self) -> None:
        index_path = os.path.join(self.persist_directory, self.INDEX_FILE)
        meta_path = os.path.join(self.persist_directory, self.META_FILE)
        self._faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        np = self._np
        vectors = np.array(embeddings, dtype="float32")
        # L2-normalise → inner product == cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = vectors / norms

        self.index.add(vectors)
        for c in chunks:
            self.metadata.append(
                {
                    "text": c["text"],
                    "source": c.get("source", "unknown"),
                    "chunk_index": int(c.get("chunk_index", 0)),
                    "strategy": c.get("strategy", "unknown"),
                }
            )

        self._persist()
        logger.info("FAISSVectorStore: added %d chunks (total=%d).", len(chunks), len(self.metadata))

    def similarity_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict]:
        np = self._np
        query = np.array([query_embedding], dtype="float32")
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []

        scores, indices = self.index.search(query, k)
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            hits.append(
                {
                    "text": meta["text"],
                    "source": meta["source"],
                    "chunk_index": meta["chunk_index"],
                    "score": round(float(score), 4),
                }
            )
        return hits

    def count(self) -> int:
        return self.index.ntotal

    def clear(self) -> None:
        import faiss
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self._persist()
        logger.warning("FAISSVectorStore: index cleared.")


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_vector_store(
    backend: str | None = None,
    **kwargs: Any,
) -> "ChromaVectorStore | FAISSVectorStore":
    """
    Return a vector store instance based on *backend* (or VECTOR_STORE_BACKEND env var).
    Pass extra keyword arguments to the store constructor (e.g. persist_directory).
    """
    backend = (backend or os.getenv("VECTOR_STORE_BACKEND", "chroma")).lower()

    if backend == "chroma":
        persist_dir = kwargs.get(
            "persist_directory",
            os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        )
        collection = kwargs.get(
            "collection_name",
            os.getenv("CHROMA_COLLECTION", "rag_collection"),
        )
        return ChromaVectorStore(
            collection_name=collection,
            persist_directory=persist_dir,
        )
    elif backend == "faiss":
        persist_dir = kwargs.get(
            "persist_directory",
            os.getenv("FAISS_PERSIST_DIR", "./faiss_db"),
        )
        dimension = int(kwargs.get("dimension", os.getenv("FAISS_DIMENSION", 1536)))
        return FAISSVectorStore(persist_directory=persist_dir, dimension=dimension)
    else:
        raise ValueError(
            f"Unknown vector store backend '{backend}'. Choose from: chroma, faiss"
        )
