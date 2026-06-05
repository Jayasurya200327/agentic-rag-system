import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # fast, good quality

VECTOR_TOP_K = 10       # fetch top 10 from vector search
BM25_TOP_K = 10         # fetch top 10 from BM25
RERANK_TOP_N = 5        # after reranking, keep top 5


# ─────────────────────────────────────────────
# STEP 1: LOAD CHROMADB COLLECTIONS
# ─────────────────────────────────────────────

def load_stores(embeddings):
    semantic_store = Chroma(
        collection_name="semantic_chunks",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    child_store = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    parent_store = Chroma(
        collection_name="parent_chunks",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return semantic_store, child_store, parent_store


# ─────────────────────────────────────────────
# STEP 2: BUILD BM25 INDEX
# ─────────────────────────────────────────────
# BM25 is a classic keyword ranking algorithm (used by Elasticsearch).
# It scores documents by term frequency + inverse document frequency.
# We build it in-memory from the same chunks stored in ChromaDB.

def build_bm25_index(store: Chroma) -> tuple[BM25Okapi, list[Document]]:
    # Pull all documents from the collection
    raw = store.get(include=["documents", "metadatas"])
    
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    # BM25 works on tokenized text (simple whitespace split is fine here)
    tokenized = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)

    print(f"  → BM25 index built: {len(docs)} documents")
    return bm25, docs


# ─────────────────────────────────────────────
# STEP 3: HYBRID RETRIEVAL
# ─────────────────────────────────────────────
# Query goes to BOTH BM25 and vector search.
# Results are merged and deduplicated.
# Then reranker rescores everything.

def hybrid_retrieve(
    query: str,
    vector_store: Chroma,
    bm25: BM25Okapi,
    bm25_docs: list[Document],
    reranker: CrossEncoder,
    use_parent_fetch: bool = False,
    parent_store: Chroma = None
) -> list[Document]:

    print(f"\n── Hybrid Retrieval ──")
    print(f"Query: '{query}'")

    # ── Vector search ──
    vector_results = vector_store.similarity_search(query, k=VECTOR_TOP_K)
    print(f"  Vector results: {len(vector_results)}")

    # ── BM25 search ──
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    # Get indices of top BM25_TOP_K scores
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]
    bm25_results = [bm25_docs[i] for i in top_bm25_indices if bm25_scores[i] > 0]
    print(f"  BM25 results: {len(bm25_results)}")

    # ── Merge + deduplicate ──
    # Use page_content as dedup key
    seen = set()
    combined = []
    for doc in vector_results + bm25_results:
        key = doc.page_content[:100]   # first 100 chars as fingerprint
        if key not in seen:
            seen.add(key)
            combined.append(doc)
    print(f"  Combined (deduped): {len(combined)}")

    # ── Rerank ──
    # CrossEncoder takes (query, document) pairs and scores each one.
    # Unlike bi-encoders (used in vector search), cross-encoders look at
    # query AND document together — much more accurate, but slower.
    # We only run it on the merged pool (not the whole corpus), so it's fast.
    reranked = rerank(query, combined, reranker)

    # ── Optional: parent fetch ──
    # If using child chunks, swap child for parent context
    if use_parent_fetch and parent_store:
        reranked = fetch_parents(reranked, parent_store)

    return reranked


# ─────────────────────────────────────────────
# STEP 4: RERANKER
# ─────────────────────────────────────────────

def rerank(
    query: str,
    docs: list[Document],
    reranker: CrossEncoder
) -> list[Document]:

    if not docs:
        return []

    # Build (query, doc_text) pairs for the cross-encoder
    pairs = [(query, doc.page_content) for doc in docs]

    # Score each pair — returns a relevance score per pair
    scores = reranker.predict(pairs)

    # Sort by score descending, keep top N
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top = scored[:RERANK_TOP_N]

    print(f"  After reranking → top {len(top)} results")
    for score, doc in top:
        print(f"    score: {score:.3f} | {doc.page_content[:80].strip()}...")

    return [doc for _, doc in top]


# ─────────────────────────────────────────────
# STEP 5: PARENT FETCH (for child-chunk results)
# ─────────────────────────────────────────────

def fetch_parents(
    child_docs: list[Document],
    parent_store: Chroma
) -> list[Document]:

    parent_ids = list(set([
        doc.metadata.get("parent_id")
        for doc in child_docs
        if doc.metadata.get("parent_id")
    ]))

    parents = []
    for pid in parent_ids:
        result = parent_store.get(where={"parent_id": pid})
        if result["documents"]:
            parents.append(Document(
                page_content=result["documents"][0],
                metadata=result["metadatas"][0] if result["metadatas"] else {}
            ))

    print(f"  Parent fetch: {len(child_docs)} children → {len(parents)} parents")
    return parents


# ─────────────────────────────────────────────
# RETRIEVER CLASS (used by LangGraph in Day 3)
# ─────────────────────────────────────────────
# Wraps everything into one clean object the graph nodes can call.

class HybridRetriever:
    def __init__(self):
        print("Initializing HybridRetriever...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Load reranker (downloads ~80MB once)
        print("Loading reranker model...")
        self.reranker = CrossEncoder(RERANK_MODEL)

        # Load ChromaDB stores
        self.semantic_store, self.child_store, self.parent_store = load_stores(self.embeddings)

        # Build BM25 index from semantic chunks
        print("Building BM25 index...")
        self.bm25_semantic, self.semantic_docs = build_bm25_index(self.semantic_store)

        # Build BM25 index from child chunks
        self.bm25_child, self.child_docs = build_bm25_index(self.child_store)

        print("HybridRetriever ready.\n")

    def retrieve_semantic(self, query: str) -> list[Document]:
        """Hybrid retrieval on semantic chunks — used for direct answers."""
        return hybrid_retrieve(
            query=query,
            vector_store=self.semantic_store,
            bm25=self.bm25_semantic,
            bm25_docs=self.semantic_docs,
            reranker=self.reranker,
            use_parent_fetch=False
        )

    def retrieve_with_parent(self, query: str) -> list[Document]:
        """Hybrid retrieval on child chunks, returns parent context."""
        return hybrid_retrieve(
            query=query,
            vector_store=self.child_store,
            bm25=self.bm25_child,
            bm25_docs=self.child_docs,
            reranker=self.reranker,
            use_parent_fetch=True,
            parent_store=self.parent_store
        )


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Agentic RAG — Day 2: Hybrid Retriever")
    print("=" * 50)

    retriever = HybridRetriever()

    queries = [
        "How does the attention mechanism work?",
        "What are the results on WMT translation tasks?",
        "multi-head attention",
    ]

    for query in queries:
        print("\n" + "=" * 50)

        # Strategy 1: semantic chunks
        print("\n[Strategy: Semantic Chunks]")
        results = retriever.retrieve_semantic(query)
        print(f"  Top result preview:")
        print(f"  {results[0].page_content[:300]}..." if results else "  No results")

        # Strategy 2: parent-child
        print("\n[Strategy: Parent-Child]")
        results = retriever.retrieve_with_parent(query)
        print(f"  Top result preview:")
        print(f"  {results[0].page_content[:300]}..." if results else "  No results")

if __name__ == "__main__":
    main()