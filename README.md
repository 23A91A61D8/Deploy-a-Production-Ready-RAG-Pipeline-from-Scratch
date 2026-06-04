# 🔍 Production-Ready RAG Pipeline from Scratch

A complete, production-ready **Retrieval-Augmented Generation (RAG)** pipeline built from the ground up — without relying on high-level abstractions like LangChain or LlamaIndex. Every component (document loading, chunking, embedding, vector storage, retrieval, and LLM generation) is implemented explicitly so the inner mechanics are fully transparent and understandable.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Setup Instructions](#-setup-instructions)
- [Running with Docker](#-running-with-docker)
- [Running Locally](#-running-locally-without-docker)
- [Configuration Reference](#-configuration-reference)
- [Chunking Strategies](#-chunking-strategies)
- [Vector Store Backends](#-vector-store-backends)
- [Embedding Backend](#-embedding-backend)
- [Example Queries & Results](#-example-queries--results)
- [Running Tests](#-running-tests)
- [Design Decisions](#-design-decisions)

---

## 🧠 Overview

Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by giving them access to external, up-to-date knowledge at inference time. Instead of relying on static training data, the model retrieves relevant document chunks and uses them as grounded context for generating answers.

**This pipeline provides:**
- ✅ Reduced hallucinations (LLM answers ONLY from provided context)
- ✅ Up-to-date knowledge (update the vector DB without retraining)
- ✅ Verifiability (source citations displayed with every answer)
- ✅ Zero external model dependencies (TF-IDF embedder needs no downloads)
- ✅ Free LLM generation via Groq API

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════╗
║              INGESTION PHASE (run once)                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Raw Documents (.txt / .md / .html)                      ║
║         │                                                ║
║         ▼                                                ║
║  [Document Loader]  ──  cleans text, removes HTML        ║
║         │                                                ║
║         ▼                                                ║
║  [Text Chunker]  ──  fixed / sentence / sliding window   ║
║         │           300-500 tokens per chunk             ║
║         ▼                                                ║
║  [Embedding Model]  ──  TF-IDF (384 dimensions)          ║
║         │                                                ║
║         ▼                                                ║
║  [Vector Store]  ──  FAISS (persisted to disk)           ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║              QUERY PHASE (per user question)             ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  User Query                                              ║
║         │                                                ║
║         ▼                                                ║
║  [Embed Query]  ──  same TF-IDF embedding model          ║
║         │                                                ║
║         ▼                                                ║
║  [Similarity Search]  ──  cosine distance, Top-K=5       ║
║         │                                                ║
║         ▼                                                ║
║  [Context Builder]  ──  format chunks + metadata         ║
║         │                                                ║
║         ▼                                                ║
║  [LLM Generation]  ──  Groq llama-3.1-8b-instant         ║
║         │                                                ║
║         ▼                                                ║
║  Answer + Source Citations                               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
rag_pipeline/
├── src/
│   ├── __init__.py
│   ├── document_loader.py   # Load and clean documents from disk
│   ├── chunker.py           # 3 chunking strategies
│   ├── embedder.py          # TF-IDF embedding (lightweight)
│   ├── vector_store.py      # ChromaDB + FAISS backends
│   ├── rag_pipeline.py      # Query-time retrieval and generation
│   └── ingest.py            # Ingestion orchestration + CLI
├── data/
│   └── documents/           # Source documents go here
│       ├── ai_overview.txt
│       ├── rag_technical_guide.txt
│       └── vector_databases.txt
├── tests/
│   ├── __init__.py
│   └── test_rag_pipeline.py # Full test suite
├── main.py                  # Application entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Document Loading** | Supports `.txt`, `.md`, `.html`, `.htm` with HTML stripping and whitespace normalization |
| **3 Chunking Strategies** | Fixed-size, Sentence-aware (default), Sliding-window |
| **Lightweight Embedder** | TF-IDF (pure Python, zero downloads, minimal RAM) |
| **2 Vector Store Backends** | FAISS (default) or ChromaDB — both with persistence |
| **Free LLM via Groq** | Uses Groq API (free tier) with llama-3.1-8b-instant |
| **Strict Prompting** | LLM answers ONLY from context; returns "I don't know" otherwise |
| **Source Citations** | Every answer shows chunk text, source file, and similarity score |
| **Fully Configurable** | Everything controlled via environment variables |
| **Dockerized** | Multi-stage Dockerfile + docker-compose.yml |
| **Test Suite** | Pytest tests covering all modules |

---

## 🛠️ Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- Groq API key (free at https://console.groq.com)

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-pipeline.git
cd rag-pipeline
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install --no-cache-dir -r requirements.txt
```

### 4. Create environment file

```bash
cp .env.example .env         # Mac/Linux
copy .env.example .env       # Windows
```

### 5. Edit .env file

Open `.env` and set your Groq API key:

```dotenv
GROQ_API_KEY=gsk_your_groq_key_here
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_BACKEND=tfidf
EMBEDDING_MODEL=tfidf-384
VECTOR_STORE_BACKEND=faiss
FAISS_DIMENSION=384
DOCS_DIR=./data/documents
CHUNKING_STRATEGY=sentence_aware
CHUNK_SIZE=400
CHUNK_OVERLAP=80
TOP_K=5
LOG_LEVEL=INFO
```

> Get a FREE Groq API key at: https://console.groq.com/signup

---

## 🐳 Running with Docker

### Build the image

```bash
docker-compose build
```

### Run ingestion

```bash
docker-compose run --rm rag-ingest
```

### Interactive Q&A

```bash
docker-compose run --rm rag-query
```

### Single query

```bash
docker-compose run --rm rag-single-query --query "What is RAG?"
```

---

## 💻 Running Locally (without Docker)

### Step 1 — Run ingestion

```bash
python main.py --ingest
```

Expected output:
```
✅ Ingestion complete:
   Documents loaded : 3
   Chunks created   : 9
   Chunks stored    : 9
```

### Step 2 — Ask a single question

```bash
python main.py --query "What is Machine Learning?"
```

### Step 3 — Interactive Q&A session

```bash
python main.py --interactive
```

### Step 4 — Debug mode (show full prompt)

```bash
python main.py --query "What is RAG?" --show_prompt
```

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Free Groq API key for LLM generation |
| `LLM_MODEL` | `llama-3.1-8b-instant` | LLM model for answer generation |
| `EMBEDDING_BACKEND` | `tfidf` | Embedding backend (tfidf or openai) |
| `EMBEDDING_MODEL` | `tfidf-384` | Embedding model name |
| `VECTOR_STORE_BACKEND` | `faiss` | Vector store (faiss or chroma) |
| `FAISS_PERSIST_DIR` | `./faiss_db` | FAISS persistence directory |
| `FAISS_DIMENSION` | `384` | Vector dimension |
| `DOCS_DIR` | `./data/documents` | Source documents directory |
| `CHUNKING_STRATEGY` | `sentence_aware` | Chunking strategy |
| `CHUNK_SIZE` | `400` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | `80` | Overlap tokens between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## ✂️ Chunking Strategies

Three strategies are implemented in `src/chunker.py`:

### `fixed_size`
Splits text every N tokens regardless of sentence boundaries. Simple baseline.

### `sentence_aware` (default)
Groups complete sentences together until chunk size is reached. Preserves semantic coherence — no sentence is ever cut in half.

### `sliding_window`
Fixed-size chunks with token-level overlap between consecutive chunks. Prevents information loss at boundaries.

---

## 🗄️ Vector Store Backends

### FAISS (default)
- Persistence: file-based (`./faiss_db/`)
- Indexing: `IndexFlatIP` with L2-normalised vectors (cosine similarity)
- Best for: all deployments, high performance

### ChromaDB
- Persistence: automatic (`./chroma_db/`)
- Indexing: HNSW with cosine similarity
- Best for: when metadata filtering is needed

---

## 🔢 Embedding Backend

### TF-IDF (default)
Pure Python implementation with zero external downloads and minimal RAM usage. Builds vocabulary from the document corpus during ingestion and uses TF-IDF weighted vectors for similarity search.

- Dimensions: 384
- Downloads required: None
- RAM required: Less than 10 MB
- Works completely offline

---

## 💬 Example Queries & Results

After ingesting the sample documents, the pipeline correctly answers:

**Q: What is Machine Learning?**
> Machine Learning (ML) is a subset of artificial intelligence that enables machines to learn from experience without being explicitly programmed. ML systems improve their performance over time as they are exposed to more data.
> Source: ai_overview.txt (chunk #0, similarity=0.4467)

**Q: What is the difference between ChromaDB and FAISS?**
> ChromaDB is easier to use with built-in filtering and persistent storage. FAISS is more powerful and scalable for large deployments but requires manual metadata management.
> Source: vector_databases.txt (chunk #1, similarity=0.4839)

**Q: What is cosine similarity?**
> Cosine similarity measures the cosine of the angle between two vectors, ranging from -1 to 1. A value of 1 means identical direction. It is scale-invariant and ideal for comparing text embeddings.
> Source: vector_databases.txt

**Q: What are chunking strategies?**
> Common strategies include fixed-size chunking, sentence-aware chunking, sliding window chunking, and recursive character text splitting.
> Source: rag_technical_guide.txt

**Q: What is a vector database?**
> A vector database stores data as high-dimensional vectors and enables efficient similarity searches across millions or billions of vectors.
> Source: vector_databases.txt

**Q: What is deep learning?**
> Deep Learning is a specialized subset of machine learning that uses neural networks with many layers. These deep neural networks can automatically learn representations from raw data.
> Source: ai_overview.txt

**Q: What are advantages of RAG over fine-tuning?**
> RAG is cheaper, supports real-time knowledge updates without retraining, provides source citations for verifiability, and reduces hallucinations by grounding answers in retrieved context.
> Source: rag_technical_guide.txt

**Q: What is HNSW indexing?**
> HNSW (Hierarchical Navigable Small World) is a graph-based indexing algorithm that builds a multi-layer graph where nearby vectors are connected. Search starts at the top layer and navigates to more specific layers.
> Source: vector_databases.txt

**Q: What is the embedding model used in RAG?**
> The embedding model converts text chunks into high-dimensional vectors. The same model must be used for both ingestion and querying to ensure vectors exist in the same semantic space.
> Source: rag_technical_guide.txt

**Q: What is RAG?**
> RAG (Retrieval-Augmented Generation) is an AI framework that enhances LLMs by allowing them to retrieve relevant information from an external knowledge base before generating a response.
> Source: rag_technical_guide.txt

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Test coverage includes:
- Document loading and HTML stripping
- All three chunking strategies
- Embedder initialization
- FAISS vector store add/search/persist/clear
- RAG pipeline prompt construction
- "I don't know" fallback behavior
- Source citation format validation

---

## 🧩 Design Decisions

### Why TF-IDF instead of neural embeddings?
Neural embedding models (HuggingFace, FastEmbed) require downloading 90MB+ model files and significant RAM to run. On resource-constrained machines, this causes out-of-memory errors. TF-IDF provides effective keyword-based similarity search with zero downloads, minimal RAM, and instant startup — making it ideal for reliable deployment across all environments.

### Why Groq instead of OpenAI?
Groq provides a 100% free API tier with no credit card required, using the llama-3.1-8b-instant model. OpenAI no longer provides free credits to new accounts. The Groq API is fully compatible with the OpenAI Python SDK, requiring only a base_url change.

### Why FAISS as default vector store?
FAISS provides excellent performance, simple file-based persistence, and works reliably without requiring a separate server process. ChromaDB is also supported for use cases requiring metadata filtering.

### Why sentence-aware chunking as default?
Sentence-aware chunking preserves semantic units. Splitting in the middle of a sentence degrades retrieval quality because the fragment becomes harder to represent meaningfully. Experiments show sentence-aware chunking consistently outperforms fixed-size chunking on retrieval recall.

### Why temperature=0 for LLM generation?
RAG is a factual Q&A system. Temperature=0 makes generation deterministic and reduces the chance of the LLM embellishing beyond the provided context.

---

## 👤 Author

**Arepalli Venkata Lakshmi**
Built as part of the Partnr GPP Program — Deploy a Production-Ready RAG Pipeline from Scratch.

---

## 📄 License

MIT License