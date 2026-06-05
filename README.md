# Agentic Document Q&A System

A production-grade RAG (Retrieval-Augmented Generation) system with an agentic 
self-correcting loop built using LangGraph, ChromaDB, and Groq.

Unlike basic RAG pipelines, this system **reasons about its own retrieval quality** — 
grading retrieved documents, detecting hallucinations, and rewriting queries when needed.

---

## Architecture

```
User Query
    │
    ▼
[Router] ── direct ──────────────────────→ [Generate] → Output
    │
  retrieve
    ▼
[Retrieve] ← ─────────────────────────── [Rewrite Query]
    │                                            ↑
    ▼                                            │
[Grade Docs] ── not relevant ───────────────────┘
    │
  relevant
    ▼
[Generate]
    │
    ▼
[Hallucination Check] ── hallucinated ──→ [Generate]
    │
  grounded
    ▼
  Output
```

---

## Key Concepts Implemented

### Advanced Chunking
- **Semantic chunking** — splits documents at meaning boundaries using embedding 
  similarity, not fixed token counts
- **Parent-child chunking** — retrieves precise small chunks, returns large parent 
  chunks as LLM context

### Hybrid Retrieval
- **BM25** (keyword search) + **ChromaDB** (vector/semantic search) combined
- Results merged, deduplicated, then reranked by a cross-encoder model
- Catches both exact keyword matches and paraphrased semantic matches

### Agentic RAG Loop (Self-RAG / Corrective RAG)
- **Router node** — decides retrieve vs direct answer before doing any retrieval
- **Retrieval grader** — LLM scores each retrieved doc for relevance, filters noise
- **Query rewriter** — rewrites failed queries and retries retrieval automatically
- **Hallucination checker** — verifies answer is grounded in retrieved context

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq (llama-3.1-8b-instant) |
| Vector DB | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Document Loading | LangChain |

---

## Project Structure

```
├── ingest.py            # Document ingestion pipeline (chunking + ChromaDB storage)
├── hybrid_retriever.py  # Hybrid BM25 + vector retriever with reranker
├── graph.py             # LangGraph agentic graph (all nodes + edges)
├── main.py              # Interactive CLI
├── data/                # Put your PDF documents here
├── chroma_db/           # Auto-generated vector store (gitignored)
├── .env.example         # Environment variable template
└── requirements.txt     # Dependencies
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-system
cd agentic-rag-system
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at: https://console.groq.com
```

### 3. Add documents and ingest

```bash
# Place PDF files in the data/ folder, then:
python ingest.py
```

### 4. Run

```bash
python main.py
```

---

## Example Interaction

```
You: What BLEU score did the Transformer achieve on WMT 2014?

[Router] → retrieve
[Retrieve] → 3 documents fetched
[Grade Docs] → 2/3 relevant
[Generate] → answer generated
[Hallucination Check] → grounded

Answer: The Transformer (big model) achieved a BLEU score of 28.4 on the 
WMT 2014 English-to-German task, outperforming all previously published 
models by more than 2.0 BLEU.
```

---

## Why This Is Better Than Basic RAG

| Basic RAG | This System |
|---|---|
| Fixed chunk sizes | Semantic + parent-child chunking |
| Vector search only | Hybrid BM25 + vector + reranker |
| Always retrieves | Router skips retrieval for general questions |
| No quality check | Retrieval grader filters irrelevant docs |
| No verification | Hallucination checker validates answers |
| Single attempt | Self-correcting requery loop |

---

## References

- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [Corrective RAG Paper](https://arxiv.org/abs/2401.15884)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)