import os
import sys
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .step-box {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #333;
    }
    .step-box.success {
        border-left-color: #28a745;
    }
    .step-box.warning {
        border-left-color: #ffc107;
    }
    .step-box.error {
        border-left-color: #dc3545;
    }
    .answer-box {
    background: #e8f4fd;
    border: 1px solid #1f77b4;
    border-radius: 8px;
    padding: 1.2rem;
    margin-top: 1rem;
    font-size: 1rem;
    line-height: 1.6;
    color: #000000 !important;   /* ← force black text */
}
    .tag {
        display: inline-block;
        background: #1f77b4;
        color: black;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        margin-right: 4px;
    }
    .tag.green { background: #28a745; }
    .tag.orange { background: #fd7e14; }
    .tag.red { background: #dc3545; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "graph" not in st.session_state:
    st.session_state.graph = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "trace_logs" not in st.session_state:
    st.session_state.trace_logs = {}


# ─────────────────────────────────────────────
# LOAD SYSTEM (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_retriever_and_graph():
    from hybrid_retriever import HybridRetriever
    from graph import build_graph
    retriever = HybridRetriever()
    graph = build_graph(retriever)
    return retriever, graph


# ─────────────────────────────────────────────
# INGEST UPLOADED PDF
# ─────────────────────────────────────────────

def ingest_uploaded_file(uploaded_file) -> dict:
    from ingest import load_documents, semantic_chunk, parent_child_chunk, store_in_chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    import shutil

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    # Save uploaded file
    save_path = os.path.join(data_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ── Close existing ChromaDB connections before deleting ──
    if st.session_state.get("retriever"):
        try:
            stores = [
                st.session_state.retriever.semantic_store,
                st.session_state.retriever.child_store,
                st.session_state.retriever.parent_store,
            ]
            for store in stores:
                store._client.close()
        except Exception:
            pass
        st.session_state.retriever = None
        st.session_state.graph = None

    # Clear cache so retriever reloads fresh
    st.cache_resource.clear()

    # Now safe to delete
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    # Run ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    docs = load_documents(data_dir)
    semantic_chunks = semantic_chunk(docs, embeddings)
    parent_chunks, child_chunks = parent_child_chunk(docs)
    store_in_chroma(semantic_chunks, parent_chunks, child_chunks, embeddings)

    st.session_state.ingested = True

    return {
        "pages": len(docs),
        "semantic_chunks": len(semantic_chunks),
        "parent_chunks": len(parent_chunks),
        "child_chunks": len(child_chunks)
    }


# ─────────────────────────────────────────────
# AGENTIC QUERY WITH TRACE
# ─────────────────────────────────────────────

def run_agentic_query(graph, query: str) -> dict:
    """Run query through graph, capture trace for display."""
    from graph import GraphState

    trace = []

    # Patch node outputs via streaming
    initial_state: GraphState = {
        "query": query,
        "rewritten_query": "",
        "documents": [],
        "generation": "",
        "retrieval_attempts": 0,
        "route": "",
        "hallucination_check": "",
        "hallucination_attempts": 0 
    }

    # Stream graph execution — captures each node's output
    final_state = None
    for step in graph.stream(initial_state):
        for node_name, node_output in step.items():
            if node_name == "router":
                route = node_output.get("route", "")
                trace.append({
                    "node": "Router",
                    "detail": f"Decision: {route.upper()}",
                    "type": "success" if route == "retrieve" else "warning"
                })
            elif node_name == "retrieve":
                attempts = node_output.get("retrieval_attempts", 1)
                doc_count = len(node_output.get("documents", []))
                trace.append({
                    "node": "Retrieve",
                    "detail": f"Fetched {doc_count} documents (attempt {attempts})",
                    "type": "success"
                })
            elif node_name == "grade_retrieval":
                doc_count = len(node_output.get("documents", []))
                trace.append({
                    "node": "Grade Retrieval",
                    "detail": f"{doc_count} relevant documents kept",
                    "type": "success" if doc_count >= 2 else "warning"
                })
            elif node_name == "rewrite_query":
                rewritten = node_output.get("rewritten_query", "")
                trace.append({
                    "node": "Rewrite Query",
                    "detail": f"New query: \"{rewritten}\"",
                    "type": "warning"
                })
            elif node_name == "generate":
                gen_len = len(node_output.get("generation", ""))
                trace.append({
                    "node": "Generate",
                    "detail": f"Answer generated ({gen_len} chars)",
                    "type": "success"
                })
            elif node_name == "check_hallucination":
                verdict = node_output.get("hallucination_check", "")
                trace.append({
                    "node": "Hallucination Check",
                    "detail": f"Verdict: {verdict.upper()}",
                    "type": "success" if verdict == "grounded" else "error"
                })
            final_state = node_output

    # Get final generation from last generate node
    generation = ""
    for step in [final_state]:
        if step and "generation" in step:
            generation = step["generation"]

    # Re-run to get full final state if generation is empty
    if not generation:
        full_final = graph.invoke(initial_state)
        generation = full_final.get("generation", "No answer generated.")

    return {"answer": generation, "trace": trace}


# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────

# Header
st.markdown('<div class="main-header">🤖 Agentic Document Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">LangGraph · Hybrid Retrieval · Self-Correcting RAG</div>', unsafe_allow_html=True)

# Two columns: sidebar-style left panel + main chat
col_left, col_right = st.columns([1, 2.5])

# ── LEFT PANEL ──
with col_left:
    st.markdown("### 📄 Document")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded:
        if st.button("Ingest Document", type="primary", use_container_width=True):
            with st.spinner("Ingesting... (chunking + embedding)"):
                try:
                    stats = ingest_uploaded_file(uploaded)
                    st.session_state.ingested = True
                    # Clear cached retriever so it reloads with new data
                    st.cache_resource.clear()
                    st.success("Ingested!")
                    st.markdown(f"""
                    **Stats:**
                    - Pages: `{stats['pages']}`
                    - Semantic chunks: `{stats['semantic_chunks']}`
                    - Parent chunks: `{stats['parent_chunks']}`
                    - Child chunks: `{stats['child_chunks']}`
                    """)
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    # Check if chroma_db exists already (from CLI ingest)
    elif os.path.exists("./chroma_db"):
        st.info("Using existing ChromaDB")
        st.session_state.ingested = True

    st.divider()

    st.markdown("### ⚙️ System")
    st.markdown("""
    **Stack:**
    - LangGraph orchestration
    - ChromaDB vector store
    - BM25 keyword search
    - Cross-encoder reranker
    - Groq LLM (llama-3.1)
    """)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.trace_logs = {}
        st.rerun()


# ── RIGHT PANEL ──
with col_right:
    st.markdown("### 💬 Ask Questions")

    # Check API key
    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set in .env file")
        st.stop()

    # Check ingestion
    if not st.session_state.ingested and not os.path.exists("./chroma_db"):
        st.warning("Upload and ingest a PDF first.")
        st.stop()

    # Load system once
    if st.session_state.graph is None:
        with st.spinner("Loading AI system..."):
            try:
                retriever, graph = load_retriever_and_graph()
                st.session_state.retriever = retriever
                st.session_state.graph = graph
            except Exception as e:
                st.error(f"Failed to load system: {e}")
                st.stop()

    # Chat history display
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.write(chat["query"])

        # Agent trace (collapsible)
        if i in st.session_state.trace_logs:
            with st.expander("🔍 Agent reasoning", expanded=False):
                for step in st.session_state.trace_logs[i]:
                    css_class = f"step-box {step['type']}"
                    st.markdown(
                        f'<div class="{css_class}"><b>{step["node"]}</b> — {step["detail"]}</div>',
                        unsafe_allow_html=True
                    )

        # Assistant answer
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="answer-box">{chat["answer"]}</div>',
                unsafe_allow_html=True
            )

    # Input box
    query = st.chat_input("Ask something about your document...")

    if query:
        # Add to history immediately
        idx = len(st.session_state.chat_history)

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = run_agentic_query(st.session_state.graph, query)
                    answer = result["answer"]
                    trace = result["trace"]
                except Exception as e:
                    answer = f"Error: {e}"
                    trace = []

            # Show trace
            if trace:
                with st.expander("🔍 Agent reasoning", expanded=True):
                    for step in trace:
                        css_class = f"step-box {step['type']}"
                        st.markdown(
                            f'<div class="{css_class}"><b>{step["node"]}</b> — {step["detail"]}</div>',
                            unsafe_allow_html=True
                        )

            # Show answer
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True
            )

        # Save to session
        st.session_state.chat_history.append({"query": query, "answer": answer})
        st.session_state.trace_logs[idx] = trace