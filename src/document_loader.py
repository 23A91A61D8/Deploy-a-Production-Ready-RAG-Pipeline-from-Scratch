"""
document_loader.py
------------------
Loads documents from a directory, extracts raw text, and performs
basic cleaning (whitespace normalization, HTML stripping, etc.).
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    clean = re.compile(r"<[^>]+>")
    return clean.sub(" ", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single ones."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_text(text: str) -> str:
    """Pipeline: strip HTML → normalize whitespace → remove non-printable chars."""
    text = _strip_html(text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)  # keep printable ASCII + newlines
    text = _normalize_whitespace(text)
    return text


def load_document(file_path: str) -> Dict[str, str]:
    """
    Read a single file and return a dict with keys:
        - 'source': the file path string
        - 'text':   the cleaned text content
    Supports: .txt, .md, .html, .htm
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    except Exception as e:
        logger.error("Failed to read %s: %s", file_path, e)
        raise

    if suffix in (".html", ".htm"):
        raw_text = _strip_html(raw_text)

    cleaned = _clean_text(raw_text)
    logger.info("Loaded '%s' — %d characters after cleaning.", path.name, len(cleaned))
    return {"source": str(path), "text": cleaned}


def load_documents_from_directory(directory: str) -> List[Dict[str, str]]:
    """
    Recursively load all supported documents (.txt, .md, .html, .htm)
    from *directory* and return a list of document dicts.
    """
    supported = {".txt", ".md", ".html", ".htm"}
    docs: List[Dict[str, str]] = []

    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    files = sorted(dir_path.rglob("*"))
    for fp in files:
        if fp.is_file() and fp.suffix.lower() in supported:
            try:
                doc = load_document(str(fp))
                if doc["text"]:          # skip empty files
                    docs.append(doc)
            except Exception as exc:
                logger.warning("Skipping %s due to error: %s", fp, exc)

    logger.info("Total documents loaded: %d", len(docs))
    return docs
