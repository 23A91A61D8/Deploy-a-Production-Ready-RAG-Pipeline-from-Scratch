"""
chunker.py
----------
Provides multiple chunking strategies:
  - Fixed-size (token-based, simple split)
  - Sentence-aware (split on sentence boundaries)
  - Sliding-window (fixed-size with overlap)

Each strategy returns a list of chunk dicts:
  {
    "text":       <str>,
    "source":     <str>,   # from the parent document
    "chunk_index":<int>,   # position in original document
  }
"""

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approximate_tokens(text: str) -> int:
    """Rough token count: split on whitespace (≈ 1 token per word)."""
    return len(text.split())


def _split_into_sentences(text: str) -> List[str]:
    """
    Naive sentence splitter using regex.
    Works well for most English prose without requiring NLTK/SpaCy.
    """
    # Split after . ? ! followed by whitespace + capital letter or end-of-string
    sentence_endings = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"])|(?<=[.?!])$")
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size chunker
# ---------------------------------------------------------------------------

def fixed_size_chunker(
    document: Dict[str, str],
    chunk_size: int = 400,
) -> List[Dict]:
    """
    Split text into non-overlapping chunks of *chunk_size* tokens each.
    """
    words = document["text"].split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            {
                "text": chunk_text,
                "source": document["source"],
                "chunk_index": len(chunks),
                "strategy": "fixed_size",
            }
        )
    logger.debug(
        "fixed_size_chunker: '%s' → %d chunks", document["source"], len(chunks)
    )
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Sentence-aware chunker
# ---------------------------------------------------------------------------

def sentence_aware_chunker(
    document: Dict[str, str],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    Group sentences into chunks that are at most *chunk_size* tokens,
    with optional *chunk_overlap* token overlap between consecutive chunks.
    """
    sentences = _split_into_sentences(document["text"])
    chunks: List[Dict] = []
    current_sentences: List[str] = []
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = _approximate_tokens(sentence)
        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            # Flush current chunk
            chunk_text = " ".join(current_sentences)
            chunks.append(
                {
                    "text": chunk_text,
                    "source": document["source"],
                    "chunk_index": chunk_index,
                    "strategy": "sentence_aware",
                }
            )
            chunk_index += 1

            # Carry over overlap sentences
            overlap_sentences: List[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tokens = _approximate_tokens(s)
                if overlap_tokens + s_tokens <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break
            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Flush remaining sentences
    if current_sentences:
        chunks.append(
            {
                "text": " ".join(current_sentences),
                "source": document["source"],
                "chunk_index": chunk_index,
                "strategy": "sentence_aware",
            }
        )

    logger.debug(
        "sentence_aware_chunker: '%s' → %d chunks", document["source"], len(chunks)
    )
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Sliding-window chunker
# ---------------------------------------------------------------------------

def sliding_window_chunker(
    document: Dict[str, str],
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Dict]:
    """
    Fixed-size chunks with a token-level sliding window overlap.
    """
    words = document["text"].split()
    stride = max(1, chunk_size - overlap)
    chunks = []
    idx = 0
    chunk_index = 0

    while idx < len(words):
        chunk_words = words[idx : idx + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            {
                "text": chunk_text,
                "source": document["source"],
                "chunk_index": chunk_index,
                "strategy": "sliding_window",
            }
        )
        chunk_index += 1
        idx += stride
        if idx >= len(words):
            break

    logger.debug(
        "sliding_window_chunker: '%s' → %d chunks", document["source"], len(chunks)
    )
    return chunks


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

STRATEGIES = {
    "fixed_size": fixed_size_chunker,
    "sentence_aware": sentence_aware_chunker,
    "sliding_window": sliding_window_chunker,
}


def chunk_documents(
    documents: List[Dict[str, str]],
    strategy: str = "sentence_aware",
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> List[Dict]:
    """
    Chunk a list of document dicts using the chosen *strategy*.

    Returns a flat list of chunk dicts across all documents.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES.keys())}"
        )

    fn = STRATEGIES[strategy]
    all_chunks: List[Dict] = []

    for doc in documents:
        if strategy == "fixed_size":
            chunks = fn(doc, chunk_size=chunk_size)
        elif strategy == "sentence_aware":
            chunks = fn(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strategy == "sliding_window":
            chunks = fn(doc, chunk_size=chunk_size, overlap=chunk_overlap)
        else:
            chunks = fn(doc)

        all_chunks.extend(chunks)

    logger.info(
        "chunk_documents: strategy='%s', total chunks=%d", strategy, len(all_chunks)
    )
    return all_chunks
