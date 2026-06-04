"""
rag_pipeline.py
---------------
Orchestrates the Query Phase of the RAG system:
  1. Embed user query
  2. Similarity search → top-K chunks
  3. Build prompt with retrieved context
  4. Call LLM → generate answer
  5. Return answer + source citations
"""

import os
import logging
from typing import List, Dict, Any

from src.embedder import Embedder
from src.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template (as specified in the project description)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are a helpful assistant designed to answer questions accurately based on a provided context. \
Your task is to use ONLY the information in the 'Context' section below.

Do not use any external knowledge you might have. If the answer to the question cannot be found \
within the provided context, you must respond with the exact phrase: "I don't know."

Context:
---
{context_block}
---

Question: {user_query}

Answer:\
"""


def _build_context_block(chunks: List[Dict]) -> str:
    """Format retrieved chunks into the context block used in the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source_label = os.path.basename(chunk.get("source", "unknown"))
        chunk_idx = chunk.get("chunk_index", "?")
        lines.append(f"[CHUNK {i}]: {chunk['text']}")
        lines.append(f"Source: {source_label} (chunk #{chunk_idx})")
        lines.append("")   # blank line between chunks
    return "\n".join(lines).strip()


def _call_openai_llm(prompt: str, model: str, max_tokens: int) -> str:
    """Call OpenAI chat completion API and return the assistant message text."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package not installed. Run: pip install openai") from exc

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,   # deterministic for RAG
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG query pipeline.

    Parameters
    ----------
    embedder       : Embedder instance (shared with ingestion!)
    vector_store   : A ChromaVectorStore or FAISSVectorStore instance
    llm_model      : LLM model name (default from LLM_MODEL env var)
    top_k          : Number of chunks to retrieve (default from TOP_K env var)
    max_tokens     : Max tokens for LLM generation
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store=None,
        llm_model: str | None = None,
        top_k: int | None = None,
        max_tokens: int = 512,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or get_vector_store()
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.top_k = top_k or int(os.getenv("TOP_K", 5))
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Step 1 – Retrieve relevant chunks
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Dict]:
        """
        Embed the query and perform similarity search.
        Returns top-K chunk dicts (text, source, chunk_index, score).
        """
        logger.info("Embedding query: '%s'", query[:80])
        query_vector = self.embedder.embed_one(query)

        logger.info("Searching vector store (top_k=%d)…", self.top_k)
        chunks = self.vector_store.similarity_search(query_vector, top_k=self.top_k)

        if not chunks:
            logger.warning("No chunks retrieved for query.")
        else:
            logger.info(
                "Retrieved %d chunks. Top score: %.4f",
                len(chunks),
                chunks[0].get("score", 0),
            )
        return chunks

    # ------------------------------------------------------------------
    # Step 2 – Build prompt
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Insert retrieved chunks and the user query into the prompt template."""
        context_block = _build_context_block(chunks)
        return PROMPT_TEMPLATE.format(
            context_block=context_block,
            user_query=query,
        )

    # ------------------------------------------------------------------
    # Step 3 – Generate answer
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Send prompt to the LLM and return the raw text answer."""
        logger.info("Calling LLM model '%s'…", self.llm_model)
        answer = _call_openai_llm(prompt, self.llm_model, self.max_tokens)
        logger.info("LLM answered (%d chars).", len(answer))
        return answer

    # ------------------------------------------------------------------
    # Full pipeline: query → answer + citations
    # ------------------------------------------------------------------

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Run the complete RAG query pipeline.

        Returns
        -------
        dict with keys:
          - "query":    the original user question
          - "answer":   the LLM-generated answer string
          - "sources":  list of source chunk dicts (text, source, chunk_index, score)
          - "prompt":   the full prompt sent to the LLM (for debugging)
        """
        if not user_query.strip():
            raise ValueError("User query cannot be empty.")

        # 1. Retrieve
        chunks = self.retrieve(user_query)

        # 2. Build prompt (even if no chunks, template instructs "I don't know")
        prompt = self.build_prompt(user_query, chunks)

        # 3. Generate
        answer = self.generate(prompt)

        return {
            "query": user_query,
            "answer": answer,
            "sources": chunks,
            "prompt": prompt,
        }
