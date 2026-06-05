import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, good quality, free

# parent chunk = large context sent to LLM
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 100

# child chunk = small, used for precise retrieval
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 30


# ─────────────────────────────────────────────
# STEP 1: LOAD DOCUMENTS
# ─────────────────────────────────────────────

def load_documents(data_dir: str) -> list[Document]:
    docs = []
    pdf_files = list(Path(data_dir).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {data_dir}/")

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        docs.extend(pages)
        print(f"  → {len(pages)} pages loaded")

    print(f"\nTotal pages loaded: {len(docs)}")
    return docs


# ─────────────────────────────────────────────
# STEP 2A: SEMANTIC CHUNKING
# ─────────────────────────────────────────────
# Splits on meaning shifts detected by embedding similarity.
# Each chunk is a coherent topic unit — no mid-sentence or mid-idea cuts.

def semantic_chunk(docs: list[Document], embeddings) -> list[Document]:
    print("\n── Semantic Chunking ──")

    # SemanticChunker compares adjacent sentences via embeddings.
    # "percentile" breakpoint: split where similarity drops below 
    # the 90th percentile (i.e. big topic shift = split point).
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )

    # Merge all page text first (page boundaries aren't semantic boundaries)
    full_text = "\n\n".join([doc.page_content for doc in docs])
    source_name = docs[0].metadata.get("source", "unknown")

    chunks = splitter.create_documents(
        [full_text],
        metadatas=[{"source": source_name, "chunk_type": "semantic"}]
    )

    print(f"  → {len(chunks)} semantic chunks created")
    print(f"  → Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    return chunks


# ─────────────────────────────────────────────
# STEP 2B: PARENT-CHILD CHUNKING
# ─────────────────────────────────────────────
# Core idea:
#   - Parent chunks (large) = what gets sent to the LLM as context
#   - Child chunks (small)  = what gets embedded and searched
#   - Each child stores parent_id in metadata → retrieval finds child,
#     then fetches parent for LLM context
#
# Why: Small chunks give precise vector matches.
#      Large chunks give rich context to the LLM.
#      Best of both worlds.

def parent_child_chunk(docs: list[Document]) -> tuple[list[Document], list[Document]]:
    print("\n── Parent-Child Chunking ──")

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    parent_chunks = []
    child_chunks = []

    # Split into parents first
    raw_parents = parent_splitter.split_documents(docs)

    for parent_id, parent in enumerate(raw_parents):
        parent.metadata["parent_id"] = f"parent_{parent_id}"
        parent.metadata["chunk_type"] = "parent"
        parent_chunks.append(parent)

        # Split each parent into children
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = f"parent_{parent_id}"
            child.metadata["chunk_type"] = "child"
            child_chunks.append(child)

    print(f"  → {len(parent_chunks)} parent chunks")
    print(f"  → {len(child_chunks)} child chunks")
    print(f"  → Avg children per parent: {len(child_chunks) // len(parent_chunks)}")
    return parent_chunks, child_chunks


# ─────────────────────────────────────────────
# STEP 3: STORE IN CHROMADB
# ─────────────────────────────────────────────
# Two collections:
#   "semantic_chunks"     → semantic chunks (queried directly)
#   "child_chunks"        → child chunks (queried, then parent fetched)
#   "parent_chunks"       → parent chunks (fetched by parent_id, NOT queried)

def store_in_chroma(
    semantic_chunks: list[Document],
    parent_chunks: list[Document],
    child_chunks: list[Document],
    embeddings
):
    print("\n── Storing in ChromaDB ──")

    # Collection 1: semantic chunks
    semantic_store = Chroma.from_documents(
        documents=semantic_chunks,
        embedding=embeddings,
        collection_name="semantic_chunks",
        persist_directory=CHROMA_DIR
    )
    print(f"  → semantic_chunks collection: {len(semantic_chunks)} docs")

    # Collection 2: child chunks (for retrieval)
    child_store = Chroma.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        collection_name="child_chunks",
        persist_directory=CHROMA_DIR
    )
    print(f"  → child_chunks collection: {len(child_chunks)} docs")

    # Collection 3: parent chunks (for context lookup — no query needed)
    # We still store them in Chroma for convenience, but retrieval 
    # will fetch by metadata filter (parent_id), not by similarity.
    parent_store = Chroma.from_documents(
        documents=parent_chunks,
        embedding=embeddings,
        collection_name="parent_chunks",
        persist_directory=CHROMA_DIR
    )
    print(f"  → parent_chunks collection: {len(parent_chunks)} docs")

    return semantic_store, child_store, parent_store


# ─────────────────────────────────────────────
# STEP 4: QUICK RETRIEVAL TEST
# ─────────────────────────────────────────────

def test_retrieval(semantic_store, child_store, parent_store, test_query: str):
    print(f"\n── Retrieval Test ──")
    print(f"Query: '{test_query}'")

    # Test 1: Semantic chunk retrieval
    print("\n[Semantic Retrieval]")
    sem_results = semantic_store.similarity_search(test_query, k=2)
    for i, doc in enumerate(sem_results):
        print(f"  Result {i+1} ({len(doc.page_content)} chars):")
        print(f"  {doc.page_content[:200]}...")

    # Test 2: Parent-child retrieval
    # Step A: find relevant child chunks
    print("\n[Parent-Child Retrieval]")
    child_results = child_store.similarity_search(test_query, k=3)

    # Step B: collect unique parent_ids from matched children
    parent_ids = list(set([doc.metadata["parent_id"] for doc in child_results]))
    print(f"  Matched child chunks: {len(child_results)}")
    print(f"  Unique parents to fetch: {parent_ids}")

    # Step C: fetch the full parent chunks by parent_id
    for pid in parent_ids[:2]:  # show first 2
        parent_results = parent_store.get(
            where={"parent_id": pid}
        )
        if parent_results["documents"]:
            parent_text = parent_results["documents"][0]
            print(f"\n  Parent chunk [{pid}] ({len(parent_text)} chars):")
            print(f"  {parent_text[:300]}...")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Agentic RAG — Day 1: Ingestion Pipeline")
    print("=" * 50)

    # Load embedding model (downloads once, ~80MB)
    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("  → Embeddings ready")

    # Load PDFs
    docs = load_documents(DATA_DIR)

    # Chunk
    semantic_chunks = semantic_chunk(docs, embeddings)
    parent_chunks, child_chunks = parent_child_chunk(docs)

    # Store
    semantic_store, child_store, parent_store = store_in_chroma(
        semantic_chunks, parent_chunks, child_chunks, embeddings
    )

    # Quick test — change this query to something relevant to your PDF
    test_query = "What is the main topic of this document?"
    test_retrieval(semantic_store, child_store, parent_store, test_query)

    print("\n✓ Day 1 complete. ChromaDB stored at ./chroma_db/")
    print("  Collections: semantic_chunks | child_chunks | parent_chunks")


if __name__ == "__main__":
    main()