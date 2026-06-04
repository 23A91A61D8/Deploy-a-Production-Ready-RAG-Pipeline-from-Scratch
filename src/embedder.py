"""
embedder.py - Ultra lightweight embedder using TF-IDF (no RAM issues)
"""
import os
import json
import math
import logging
from typing import List

logger = logging.getLogger(__name__)

class TFIDFEmbedder:
    """
    Pure Python TF-IDF embedder - needs ZERO downloads, ZERO RAM!
    Works completely offline with no external models.
    """
    def __init__(self, vocab_size: int = 384):
        self.vocab_size = vocab_size
        self.vocab: dict = {}
        self.idf: dict = {}
        self.fitted = False
        vocab_path = os.getenv("TFIDF_VOCAB_PATH", "./faiss_db/tfidf_vocab.json")
        if os.path.exists(vocab_path):
            self._load_vocab(vocab_path)

    def _tokenize(self, text: str) -> List[str]:
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z]{2,}\b', text)
        return tokens

    def _save_vocab(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "idf": self.idf}, f)

    def _load_vocab(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.fitted = True
        logger.info("TFIDFEmbedder: loaded vocab (%d terms)", len(self.vocab))

    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        # Count document frequency
        df: dict = {}
        all_tokens = []
        for text in texts:
            tokens = set(self._tokenize(text))
            all_tokens.append(tokens)
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        # Keep top vocab_size terms by document frequency
        sorted_terms = sorted(df.items(), key=lambda x: x[1], reverse=True)
        top_terms = [t for t, _ in sorted_terms[:self.vocab_size]]
        self.vocab = {term: idx for idx, term in enumerate(top_terms)}

        # Compute IDF
        N = len(texts)
        self.idf = {
            term: math.log((N + 1) / (df.get(term, 0) + 1)) + 1
            for term in self.vocab
        }
        self.fitted = True

        # Save vocab
        vocab_path = os.getenv("TFIDF_VOCAB_PATH", "./faiss_db/tfidf_vocab.json")
        self._save_vocab(vocab_path)
        logger.info("TFIDFEmbedder: fitted vocab size=%d", len(self.vocab))

    def _embed_one(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        tf: dict = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = max(len(tokens), 1)

        vector = [0.0] * self.vocab_size
        for term, idx in self.vocab.items():
            if term in tf:
                tfidf = (tf[term] / total) * self.idf.get(term, 1.0)
                vector[idx] = tfidf

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.fitted:
            logger.warning("TFIDFEmbedder not fitted! Fitting on input texts.")
            self.fit(texts)
        return [self._embed_one(t) for t in texts]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class Embedder:
    def __init__(self, backend: str = None, model: str = None):
        self.backend = "tfidf"
        self.model = "tfidf-384"
        self._tfidf = TFIDFEmbedder(vocab_size=384)
        logger.info("Embedder initialized: backend=tfidf (lightweight)")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._tfidf.embed(texts)

    def embed_one(self, text: str) -> List[float]:
        return self._tfidf.embed_one(text)