# Production RAG Fundamentals

A modular, production-focused codebase implementing core Retrieval-Augmented Generation (RAG) concepts with **LangChain**, **ChromaDB**, **PostgreSQL (pgvector)**, and **LangSmith**.

This repository follows the curriculum from the freeCodeCamp course:  
**[Production RAG with LangChain & Vector Databases – Full Course](http://www.youtube.com/watch?v=mHxLXzYjQRE)**.

---

## Architecture Overview

The codebase is organized into sequential modules reflecting a production RAG lifecycle:

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion & Preprocessing"]
        A[Documents / PDF / Text / Web] -->|PyPDFLoader / TextLoader| B[Raw Documents]
        B -->|Recursive / Semantic Chunker| C[Document Chunks]
        C -->|OpenAI / Titan / Qwen Embeddings| D[Vector Embeddings]
    end

    subgraph Storage["2. Storage & Indexing"]
        D -->|Store Vectors| E[ChromaDB Local]
        D -->|Store Vectors| F[Supabase / pgvector Cloud]
        C -->|Tokenize Text| G[BM25 Inverted Index]
    end

    subgraph Retrieval["3. Advanced Retrieval"]
        Q[User Query] --> H{Retrieval Strategy}
        H -->|Dense + Sparse| I[Hybrid Search: Weighted RRF / Ensemble]
        H -->|Query Expansion| J[Multi-Query Retriever]
        H -->|Context Preserving| K[Parent Document Retriever]
        H -->|Noise Reduction| L[Contextual Compression]
        E --> I
        F --> I
        G --> I
    end

    subgraph Generation["4. Generation & Pipelines"]
        I --> M[LCEL RAG Chain]
        J --> M
        K --> M
        L --> M
        M -->|Context + Prompt| N[LLM Generation: OpenRouter]
        N --> O[Responses: Grounded Answers, Sources, Structured Output]
    end

    subgraph Observability["5. Production Observability"]
        N -.->|Tracing & Latency| P[LangSmith Tracing]
        N -.->|Complexity Routing| Q1[Model Router]
        N -.->|Exact / Semantic Cache| Q2[Semantic Cache]
        N -.->|JSON Logs & Metrics| Q3[Metrics Collector]
    end
```

---

## Tech Stack

- **Orchestration:** LangChain (`langchain`, `langchain-core`, `langchain-community`, `langchain-classic`, `langchain-experimental`, `langgraph`)
- **Vector Databases:** ChromaDB (`chromadb`, `langchain-chroma`), PostgreSQL + pgvector (`langchain-postgres`)
- **Sparse Retrieval:** BM25 (`rank-bm25`)
- **LLM & Embedding Providers:** OpenRouter / OpenAI API (`langchain-openai`)
- **Observability:** LangSmith (`langsmith`)
- **Document Parsing:** PyPDF (`pypdf`)
- **Package Management:** `uv` / Python `>=3.13`

---

## Repository Structure

```text
rag-course/
├── 00-setup/
│   └── verify_setup.py          # Environment verification & OpenRouter LLM test invoke
├── 01-document-loaders/
│   └── document_loaders.py      # PDF, text, web, and directory loaders with metadata handling
├── 02-chunking/
│   ├── text_splitters.py        # RecursiveCharacterTextSplitter and chunk size/overlap analysis
│   ├── semantic_chunking.py     # SemanticChunker vs recursive splitting with topic preservation
│   └── prod_ready.py            # Production-grade smart chunker with recursive fallback
├── 03-embeddings/
│   ├── raw_emb.py               # Direct low-level OpenAI client embedding requests
│   └── embeddings_deep.py       # Batching, cosine similarity, norms, and CacheBackedEmbeddings
├── 04-vector-stores/
│   ├── app.py                   # Bare-metal ChromaDB client collection and query operations
│   ├── vector_stores.py         # LangChain Chroma: similarity with score, metadata filter, MMR, disk persistence
│   └── supabase_connection.py   # Cloud vector storage integration using Supabase pgvector
├── 05-retrieval/
│   ├── hybrid_search.py         # Custom Weighted Reciprocal Rank Fusion (RRF) combining BM25 & Chroma
│   ├── final_production.py      # LangChain EnsembleRetriever benchmarked on challenging keywords
│   └── advanced_rag.py          # MultiQueryRetriever, ContextualCompressionRetriever, & ParentDocumentRetriever
├── 06-pipelines/
│   ├── rag_pipeline.py          # LCEL chains: standard RAG, source attribution, fallback handling, structured output
│   └── production_example.py    # Production RAGService class backed by PGVector with scoring & filtering
├── 07-observability/
│   ├── langsmith_setup.py       # Tracing setup with @traceable, tags, and run metadata
│   ├── cost_optimization.py     # ModelRouter (complexity routing) and SemanticCache for cost reduction
│   └── monitoring.py            # JSON structured logging, MetricsCollector, and InstrumentedLLM
├── docs/
│   └── langchain_demo.pdf       # Sample PDF document for loader benchmarks
├── pyproject.toml               # Project metadata and locked dependency specifications
└── .gitignore                   # Excludes .env, local Chroma databases, caches, and virtual environments
```

---

## Prerequisites & Setup Instructions

### 1. Requirements
- Python `3.13+`
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip` / `venv`

### 2. Clone and Install Dependencies

Using `uv` (recommended):
```bash
# Clone the repository
git clone https://github.com/production-rag-course/rag-fundamentals.git
cd rag-course

# Create virtual environment and install dependencies from lockfile
uv sync
```

Or using standard `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Environment Variables Configuration

Create a `.env` file in the project root:

```env
# Primary LLM & Embedding provider endpoint (OpenRouter)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Observability (Optional, for LangSmith tracing)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=rag-fundamentals

# Database (Optional, for pgvector / Supabase modules in 04 & 06)
SUPABASE_DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

---

## How to Run

Activate your virtual environment before running scripts:
```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### Module 00: Environment Verification
```bash
python 00-setup/verify_setup.py
```

### Module 01: Document Loaders
```bash
python 01-document-loaders/document_loaders.py
```

### Module 02: Chunking Strategies
```bash
# Recursive splitting and chunk overlap benchmark
python 02-chunking/text_splitters.py

# Semantic chunking vs recursive splitting comparison
python 02-chunking/semantic_chunking.py

# Production chunker with recursive fallback
python 02-chunking/prod_ready.py
```

### Module 03: Embeddings Deep Dive
```bash
# Raw embedding generation via client
python 03-embeddings/raw_emb.py

# Cosine similarity and CacheBackedEmbeddings
python 03-embeddings/embeddings_deep.py
```

### Module 04: Vector Stores
```bash
# In-memory ChromaDB native client
python 04-vector-stores/app.py

# Chroma similarity search, metadata filtering, MMR, and disk persistence
python 04-vector-stores/vector_stores.py

# Supabase pgvector connection test (requires SUPABASE_DATABASE_URL)
python 04-vector-stores/supabase_connection.py
```

### Module 05: Advanced Retrieval
```bash
# Custom Weighted Reciprocal Rank Fusion (BM25 + Dense)
python 05-retrieval/hybrid_search.py

# LangChain EnsembleRetriever benchmark
python 05-retrieval/final_production.py

# Multi-Query, Contextual Compression, and Parent Document Retriever
python 05-retrieval/advanced_rag.py
```

### Module 06: End-to-End RAG Pipelines
```bash
# LCEL RAG chains with sources, fallback, and Pydantic structured output
python 06-pipelines/rag_pipeline.py

# Production RAGService with PGVector lazy loading and source scoring
python 06-pipelines/production_example.py
```

### Module 07: Observability & Cost Optimization
```bash
# LangSmith tracing and run tagging
python 07-observability/langsmith_setup.py

# Dynamic ModelRouter (cheap vs expensive) and SemanticCache
python 07-observability/cost_optimization.py

# Structured JSON logging and latency/token MetricsCollector
python 07-observability/monitoring.py
```

---

## Key Concepts & Patterns Implemented

| Pattern | Script | Description |
| :--- | :--- | :--- |
| **Semantic Chunking** | `02-chunking/semantic_chunking.py` | Uses embedding distance percentiles to detect natural topic transitions, preventing fragmented context. |
| **Resilient Chunking** | `02-chunking/prod_ready.py` | Runs semantic chunking with automated fallback to recursive splitting for oversized text chunks. |
| **Embedding Caching** | `03-embeddings/embeddings_deep.py` | Implements `CacheBackedEmbeddings` with a byte store to avoid redundant API embedding calls. |
| **MMR & Filtering** | `04-vector-stores/vector_stores.py` | Maximal Marginal Relevance balances relevance with diversity; metadata filters constrain search space. |
| **Hybrid RRF** | `05-retrieval/hybrid_search.py` | Weighted Reciprocal Rank Fusion fuses lexical (BM25) and semantic vector ranks for superior retrieval on keywords, acronyms, and product codes. |
| **Context Compression** | `05-retrieval/advanced_rag.py` | Uses `LLMChainExtractor` to strip irrelevant sentences from retrieved chunks before context injection. |
| **Parent Document** | `05-retrieval/advanced_rag.py` | Embeds smaller child chunks for precise retrieval matching while returning full parent documents for generation. |
| **Structured Output** | `06-pipelines/rag_pipeline.py` | Enforces type-safe Pydantic models (`RAGResponse`) containing answers, confidence ratings, and source lists. |
| **Model Routing** | `07-observability/cost_optimization.py` | Classifies query complexity to route simple queries to fast/cheap models and complex queries to larger reasoning models. |
| **Structured Metrics** | `07-observability/monitoring.py` | Real-time tracking of request counts, error rates, token usage, cache hit ratio, and p99 latencies via JSON logs. |
