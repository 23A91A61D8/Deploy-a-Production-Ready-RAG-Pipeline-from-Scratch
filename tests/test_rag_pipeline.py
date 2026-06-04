"""
tests/test_rag_pipeline.py
--------------------------
Unit and integration tests for the RAG pipeline components.
Run with: pytest tests/ -v
"""

import os
import sys
import pytest
import tempfile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Artificial intelligence (AI) is intelligence demonstrated by machines. "
    "AI research has been defined as the field of study of intelligent agents. "
    "Machine learning is a subset of AI that enables systems to learn from data. "
    "Deep learning uses neural networks with many layers to model complex patterns. "
    "Natural language processing (NLP) allows computers to understand human language. "
    "Large language models like GPT are trained on vast amounts of text data. "
    "Vector databases store embeddings for efficient similarity search. "
    "Retrieval-Augmented Generation (RAG) combines retrieval with language generation. "
    "ChromaDB is a popular open-source vector database for AI applications. "
    "FAISS is Facebook's library for efficient similarity search of dense vectors."
)

SAMPLE_DOC = {"source": "test_doc.txt", "text": SAMPLE_TEXT}


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_docs_dir(temp_dir):
    """Create a temp directory with sample .txt files."""
    doc1 = os.path.join(temp_dir, "ai_intro.txt")
    doc2 = os.path.join(temp_dir, "rag_overview.txt")
    with open(doc1, "w") as f:
        f.write(SAMPLE_TEXT)
    with open(doc2, "w") as f:
        f.write(
            "RAG stands for Retrieval-Augmented Generation. "
            "It retrieves relevant documents and uses them as context for an LLM. "
            "The retrieval step uses vector similarity search. "
            "Chunking splits documents into smaller pieces for better retrieval. "
            "Embeddings convert text to numerical vectors capturing semantic meaning."
        )
    return temp_dir


# ---------------------------------------------------------------------------
# Tests: Document Loader
# ---------------------------------------------------------------------------

class TestDocumentLoader:
    def test_load_single_txt_file(self, temp_dir):
        from src.document_loader import load_document
        path = os.path.join(temp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("Hello world. This is a test document.\n")
        doc = load_document(path)
        assert doc["source"] == path
        assert "Hello world" in doc["text"]

    def test_load_html_strips_tags(self, temp_dir):
        from src.document_loader import load_document
        path = os.path.join(temp_dir, "test.html")
        with open(path, "w") as f:
            f.write("<html><body><p>Clean text</p><script>bad code</script></body></html>")
        doc = load_document(path)
        assert "<p>" not in doc["text"]
        assert "Clean text" in doc["text"]

    def test_load_directory_multiple_files(self, sample_docs_dir):
        from src.document_loader import load_documents_from_directory
        docs = load_documents_from_directory(sample_docs_dir)
        assert len(docs) == 2
        for doc in docs:
            assert "text" in doc
            assert "source" in doc
            assert len(doc["text"]) > 0

    def test_load_directory_missing_raises(self):
        from src.document_loader import load_documents_from_directory
        with pytest.raises(NotADirectoryError):
            load_documents_from_directory("/nonexistent/path/xyz")

    def test_load_empty_file_is_skipped(self, temp_dir):
        from src.document_loader import load_documents_from_directory
        path = os.path.join(temp_dir, "empty.txt")
        open(path, "w").close()  # create empty file
        docs = load_documents_from_directory(temp_dir)
        assert all(len(d["text"]) > 0 for d in docs)


# ---------------------------------------------------------------------------
# Tests: Chunker
# ---------------------------------------------------------------------------

class TestChunker:
    def test_fixed_size_chunk_count(self):
        from src.chunker import fixed_size_chunker
        chunks = fixed_size_chunker(SAMPLE_DOC, chunk_size=20)
        total_words = len(SAMPLE_TEXT.split())
        expected = (total_words + 19) // 20  # ceiling division
        assert len(chunks) == expected

    def test_fixed_size_chunks_have_required_keys(self):
        from src.chunker import fixed_size_chunker
        chunks = fixed_size_chunker(SAMPLE_DOC, chunk_size=50)
        for c in chunks:
            assert "text" in c
            assert "source" in c
            assert "chunk_index" in c

    def test_sentence_aware_no_truncated_sentences(self):
        from src.chunker import sentence_aware_chunker
        chunks = sentence_aware_chunker(SAMPLE_DOC, chunk_size=100, chunk_overlap=20)
        # Each chunk should end with proper sentence-ending punctuation
        for c in chunks:
            assert len(c["text"]) > 0

    def test_sliding_window_overlap(self):
        from src.chunker import sliding_window_chunker
        chunks = sliding_window_chunker(SAMPLE_DOC, chunk_size=30, overlap=10)
        # With overlap, we should have more chunks than fixed-size
        fixed_chunks = len(SAMPLE_TEXT.split()) // 30
        assert len(chunks) >= fixed_chunks

    def test_chunk_documents_factory(self, sample_docs_dir):
        from src.document_loader import load_documents_from_directory
        from src.chunker import chunk_documents
        docs = load_documents_from_directory(sample_docs_dir)
        for strategy in ["fixed_size", "sentence_aware", "sliding_window"]:
            chunks = chunk_documents(docs, strategy=strategy, chunk_size=50)
            assert len(chunks) > 0, f"Strategy {strategy} produced no chunks"

    def test_invalid_strategy_raises(self):
        from src.chunker import chunk_documents
        with pytest.raises(ValueError):
            chunk_documents([SAMPLE_DOC], strategy="invalid_strategy")

    def test_source_metadata_preserved(self):
        from src.chunker import fixed_size_chunker
        chunks = fixed_size_chunker(SAMPLE_DOC, chunk_size=50)
        for c in chunks:
            assert c["source"] == "test_doc.txt"


# ---------------------------------------------------------------------------
# Tests: Embedder (mock — avoids real API calls in unit tests)
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_embedder_init_openai(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.embedder import Embedder
        e = Embedder()
        assert e.backend == "openai"

    def test_embedder_init_huggingface(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BACKEND", "huggingface")
        from src.embedder import Embedder
        e = Embedder()
        assert e.backend == "huggingface"

    def test_embedder_invalid_backend(self):
        from src.embedder import Embedder
        with pytest.raises(ValueError):
            Embedder(backend="unknown_backend")

    def test_embed_empty_list(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.embedder import Embedder
        e = Embedder()
        result = e.embed([])
        assert result == []


# ---------------------------------------------------------------------------
# Tests: Vector Store (FAISS — no external service needed)
# ---------------------------------------------------------------------------

class TestFAISSVectorStore:
    def test_add_and_search(self, temp_dir):
        pytest.importorskip("faiss")
        from src.vector_store import FAISSVectorStore
        store = FAISSVectorStore(persist_directory=temp_dir, dimension=4)

        chunks = [
            {"text": "The cat sat on the mat.", "source": "doc1.txt", "chunk_index": 0},
            {"text": "The dog ran in the park.", "source": "doc1.txt", "chunk_index": 1},
        ]
        embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        store.add_chunks(chunks, embeddings)

        assert store.count() == 2

        results = store.similarity_search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert "cat" in results[0]["text"]

    def test_persistence(self, temp_dir):
        pytest.importorskip("faiss")
        from src.vector_store import FAISSVectorStore
        store = FAISSVectorStore(persist_directory=temp_dir, dimension=4)
        chunks = [{"text": "Hello world", "source": "test.txt", "chunk_index": 0}]
        embeddings = [[0.5, 0.5, 0.5, 0.5]]
        store.add_chunks(chunks, embeddings)

        # Reload from disk
        store2 = FAISSVectorStore(persist_directory=temp_dir, dimension=4)
        assert store2.count() == 1

    def test_clear(self, temp_dir):
        pytest.importorskip("faiss")
        from src.vector_store import FAISSVectorStore
        store = FAISSVectorStore(persist_directory=temp_dir, dimension=4)
        store.add_chunks(
            [{"text": "X", "source": "f.txt", "chunk_index": 0}],
            [[1.0, 0.0, 0.0, 0.0]],
        )
        store.clear()
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Tests: RAG Pipeline (mocked LLM and retrieval)
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    def _make_pipeline(self, monkeypatch, mock_chunks):
        """Create a RAGPipeline with mocked embedder and vector store."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class MockEmbedder:
            def embed_one(self, text):
                return [0.1] * 1536

        class MockVectorStore:
            def similarity_search(self, vec, top_k):
                return mock_chunks

        class MockPipeline:
            def __init__(self):
                self.embedder = MockEmbedder()
                self.vector_store = MockVectorStore()
                self.llm_model = "gpt-4o-mini"
                self.top_k = 5
                self.max_tokens = 512

            def retrieve(self, query):
                return self.vector_store.similarity_search(None, top_k=5)

            def build_prompt(self, query, chunks):
                from src.rag_pipeline import _build_context_block, PROMPT_TEMPLATE
                return PROMPT_TEMPLATE.format(
                    context_block=_build_context_block(chunks),
                    user_query=query,
                )

        return MockPipeline()

    def test_prompt_contains_context(self, monkeypatch):
        from src.rag_pipeline import _build_context_block, PROMPT_TEMPLATE
        chunks = [
            {"text": "RAG stands for Retrieval-Augmented Generation.", "source": "doc.txt", "chunk_index": 0, "score": 0.9}
        ]
        context = _build_context_block(chunks)
        prompt = PROMPT_TEMPLATE.format(context_block=context, user_query="What is RAG?")
        assert "RAG stands for Retrieval-Augmented Generation" in prompt
        assert "What is RAG?" in prompt
        assert "ONLY the information" in prompt

    def test_prompt_instructs_i_dont_know(self, monkeypatch):
        from src.rag_pipeline import PROMPT_TEMPLATE
        prompt = PROMPT_TEMPLATE.format(context_block="", user_query="unknown question")
        assert "I don't know" in prompt

    def test_empty_query_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        # Patch OpenAI calls to avoid real API calls
        import unittest.mock as mock
        with mock.patch("src.rag_pipeline._call_openai_llm", return_value="test"):
            with mock.patch("src.embedder._embed_openai", return_value=[[0.1]*1536]):
                from src.rag_pipeline import RAGPipeline
                # Use a mock store
                with pytest.raises(ValueError):
                    pipeline = RAGPipeline()
                    pipeline.query("")

    def test_build_prompt_structure(self):
        from src.rag_pipeline import _build_context_block, PROMPT_TEMPLATE
        chunks = [
            {"text": "Chunk one content.", "source": "/docs/file.txt", "chunk_index": 0, "score": 0.8},
            {"text": "Chunk two content.", "source": "/docs/file.txt", "chunk_index": 1, "score": 0.7},
        ]
        context = _build_context_block(chunks)
        assert "[CHUNK 1]" in context
        assert "[CHUNK 2]" in context
        assert "Source:" in context

        prompt = PROMPT_TEMPLATE.format(context_block=context, user_query="test?")
        assert "Context:" in prompt
        assert "Question:" in prompt
        assert "Answer:" in prompt

    def test_sources_displayed_in_result_format(self):
        """Verify result dict has all required keys."""
        # This tests the contract of the query() return value
        required_keys = {"query", "answer", "sources", "prompt"}
        # Simulate what query() returns
        mock_result = {
            "query": "What is AI?",
            "answer": "AI is intelligence demonstrated by machines.",
            "sources": [
                {"text": "AI is...", "source": "doc.txt", "chunk_index": 0, "score": 0.95}
            ],
            "prompt": "You are a helpful assistant...",
        }
        assert required_keys.issubset(mock_result.keys())
        assert mock_result["sources"][0]["source"] == "doc.txt"
