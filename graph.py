import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from groq import Groq
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from hybrid_retriever import HybridRetriever

load_dotenv()

# ─────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

def call_llm(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
# LangGraph passes this dict between every node.
# Each node reads from it and writes back to it.

class GraphState(TypedDict):
    query: str                        # original user question
    rewritten_query: str              # rewritten query (if retrieval was bad)
    documents: list[Document]         # retrieved documents
    generation: str                   # LLM's answer
    retrieval_attempts: int           # how many times we've tried retrieval
    route: str                        # "retrieve" or "direct"
    hallucination_check: str          # "grounded" or "hallucinated"
    hallucination_attempts: int       # how many times we've tried regenerating due to hallucination

# ─────────────────────────────────────────────
# NODE 1: ROUTER
# ─────────────────────────────────────────────
# Decides whether the question needs document retrieval or
# can be answered directly from LLM knowledge.
# 
# Examples that go "direct":
#   - "What is 2+2?"
#   - "Explain what a neural network is"
# Examples that go "retrieve":
#   - "What BLEU score did the Transformer achieve?"
#   - "What dropout rate was used in the base model?"

def router_node(state: GraphState) -> GraphState:
    print("\n[NODE: Router]")

    system = """You are a routing assistant. 
Decide if the question needs retrieval from documents or can be answered directly.

Reply with ONLY one word:
- "retrieve" if the question needs specific facts, numbers, or details from documents
- "direct" if the question is general knowledge that doesn't need documents"""

    user = f"Question: {state['query']}"
    decision = call_llm(system, user).lower()

    # Normalize — LLM might say "retrieve." or "direct answer"
    if "direct" in decision:
        route = "direct"
    else:
        route = "retrieve"

    print(f"  Decision: {route}")
    return {**state, "route": route}


# ─────────────────────────────────────────────
# NODE 2: RETRIEVE
# ─────────────────────────────────────────────
# Runs hybrid retrieval using Day 2's HybridRetriever.
# Uses rewritten_query if available (after a failed retrieval),
# otherwise uses the original query.

def retrieve_node(state: GraphState, retriever: HybridRetriever) -> GraphState:
    print("\n[NODE: Retrieve]")

    # Use rewritten query if we're on a retry
    query = state.get("rewritten_query") or state["query"]
    attempts = state.get("retrieval_attempts", 0) + 1

    print(f"  Query: '{query}' (attempt {attempts})")

    # Parent-child gives better results for specific questions
    docs = retriever.retrieve_with_parent(query)

    print(f"  Retrieved {len(docs)} documents")
    return {**state, "documents": docs, "retrieval_attempts": attempts}


# ─────────────────────────────────────────────
# NODE 3: GRADE RETRIEVAL
# ─────────────────────────────────────────────
# LLM checks each retrieved document for relevance.
# Filters out irrelevant docs.
# If too few relevant docs remain → triggers rewrite loop.

def grade_retrieval_node(state: GraphState) -> GraphState:
    print("\n[NODE: Grade Retrieval]")

    query = state.get("rewritten_query") or state["query"]
    docs = state["documents"]
    relevant_docs = []

    system = """You are a relevance grader.
Given a question and a document, decide if the document contains information useful for answering the question.

Reply with ONLY one word: "relevant" or "irrelevant"."""

    for i, doc in enumerate(docs):
        user = f"""Question: {query}

Document:
{doc.page_content[:500]}"""

        verdict = call_llm(system, user).lower()
        is_relevant = "relevant" in verdict and "irrelevant" not in verdict

        print(f"  Doc {i+1}: {verdict.strip()} | {doc.page_content[:60].strip()}...")
        if is_relevant:
            relevant_docs.append(doc)

    print(f"  Relevant: {len(relevant_docs)}/{len(docs)}")
    return {**state, "documents": relevant_docs}


# ─────────────────────────────────────────────
# NODE 4: REWRITE QUERY
# ─────────────────────────────────────────────
# If retrieval grading found too few relevant docs,
# the LLM rewrites the query to be more specific/searchable.
# Then the graph loops back to retrieve again.

def rewrite_query_node(state: GraphState) -> GraphState:
    print("\n[NODE: Rewrite Query]")

    system = """You are a query rewriting assistant.
The original query failed to retrieve relevant documents.
Rewrite it to be more specific and likely to match relevant content.
Reply with ONLY the rewritten query, nothing else."""

    user = f"Original query: {state['query']}"
    rewritten = call_llm(system, user)

    print(f"  Original : {state['query']}")
    print(f"  Rewritten: {rewritten}")
    return {**state, "rewritten_query": rewritten}


# ─────────────────────────────────────────────
# NODE 5: GENERATE
# ─────────────────────────────────────────────
# Generates the final answer using retrieved documents as context.
# For "direct" route, answers without any document context.

def generate_node(state: GraphState) -> GraphState:
    print("\n[NODE: Generate]")

    query = state.get("rewritten_query") or state["query"]
    docs = state.get("documents", [])
    route = state.get("route", "retrieve")

    if route == "direct":
        # Router decided no retrieval needed — use LLM knowledge freely
        system = "You are a helpful assistant. Answer the question clearly and concisely."
        user = query

    elif docs:
        # Retrieval succeeded — answer strictly from context
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        system = """You are a helpful assistant that answers questions ONLY based on provided context.
If the context does not contain enough information to answer the question, say:
"I couldn't find relevant information in the document to answer this question."
Do NOT use your own knowledge. Stick strictly to the context."""
        user = f"""Context:
{context}

Question: {query}

Answer:"""
        
    else:
        # Retrieval was attempted but all docs were graded irrelevant
        system = "You are a helpful assistant."
        user = f"""The document did not contain relevant information to answer this question.
Inform the user clearly: "I couldn't find relevant information in the document to answer this question."
Question: {query}"""

    answer = call_llm(system, user)
    print(f"  Generated answer ({len(answer)} chars)")
    return {**state, "generation": answer}

# ─────────────────────────────────────────────
# NODE 6: CHECK HALLUCINATION
# ─────────────────────────────────────────────
# Verifies that the generated answer is grounded in the retrieved docs.
# If the LLM made something up that isn't in the context → "hallucinated"
# → graph loops back to generate again.

def check_hallucination_node(state: GraphState) -> GraphState:
    print("\n[NODE: Check Hallucination]")

    docs = state.get("documents", [])
    generation = state["generation"]
    attempts = state.get("hallucination_attempts", 0) + 1

    if not docs:
        print("  No docs to check (direct route) → grounded")
        return {**state, "hallucination_check": "grounded", "hallucination_attempts": attempts}

    # Safety valve — after 2 attempts just accept the answer
    if attempts >= 2:
        print(f"  Max hallucination attempts reached → accepting answer")
        return {**state, "hallucination_check": "grounded", "hallucination_attempts": attempts}

    context = "\n\n".join([doc.page_content[:500] for doc in docs])  # increased from 300

    system = """You are a hallucination detector.
Given a context and an answer, decide if the answer is grounded.

Rules:
- If the answer is based on information present in the context, reply "grounded"
- Only reply "hallucinated" if the answer contains specific claims clearly NOT in the context
- Paraphrasing and summarizing the context counts as grounded

Reply with ONLY one word: "grounded" or "hallucinated"."""

    user = f"""Context:
{context}

Answer:
{generation}"""

    verdict = call_llm(system, user).lower()
    result = "grounded" if "grounded" in verdict else "hallucinated"

    print(f"  Verdict: {result} (attempt {attempts})")
    return {**state, "hallucination_check": result, "hallucination_attempts": attempts}

# ─────────────────────────────────────────────
# CONDITIONAL EDGE FUNCTIONS
# ─────────────────────────────────────────────
# These functions decide which node to go to next.
# LangGraph calls them after each node and routes based on return value.

def route_after_router(state: GraphState) -> Literal["retrieve", "generate"]:
    return "retrieve" if state["route"] == "retrieve" else "generate"


def route_after_grading(state: GraphState) -> Literal["generate", "rewrite"]:
    relevant_count = len(state["documents"])
    attempts = state.get("retrieval_attempts", 0)

    # If we have at least 2 relevant docs → generate
    # If not, and we haven't retried too many times → rewrite
    if relevant_count >= 2:
        print(f"  → Routing to: generate ({relevant_count} relevant docs)")
        return "generate"
    elif attempts >= 2:
        # Safety valve — avoid infinite loop
        print(f"  → Max attempts reached, generating with what we have")
        return "generate"
    else:
        print(f"  → Routing to: rewrite (only {relevant_count} relevant docs)")
        return "rewrite"


def route_after_hallucination(state: GraphState) -> Literal["generate", "end"]:
    check = state.get("hallucination_check", "grounded")
    attempts = state.get("hallucination_attempts", 0)
    if check == "hallucinated" and attempts < 2:
        print("  → Routing to: regenerate")
        return "generate"
    else:
        print("  → Routing to: end")
        return "end"


# ─────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_graph(retriever: HybridRetriever) -> StateGraph:

    # Wrap nodes that need retriever via closure
    def retrieve(state): return retrieve_node(state, retriever)

    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_retrieval", grade_retrieval_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("check_hallucination", check_hallucination_node)

    # Entry point
    workflow.set_entry_point("router")

    # Conditional edge: router → retrieve OR generate
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {"retrieve": "retrieve", "generate": "generate"}
    )

    # Fixed edge: retrieve → grade
    workflow.add_edge("retrieve", "grade_retrieval")

    # Conditional edge: grade → generate OR rewrite
    workflow.add_conditional_edges(
        "grade_retrieval",
        route_after_grading,
        {"generate": "generate", "rewrite": "rewrite_query"}
    )

    # Fixed edge: rewrite → retrieve (the loop)
    workflow.add_edge("rewrite_query", "retrieve")

    # Fixed edge: generate → check hallucination
    workflow.add_edge("generate", "check_hallucination")

    # Conditional edge: hallucination check → regenerate OR end
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {"generate": "generate", "end": END}
    )

    return workflow.compile()


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

def run_query(graph, query: str) -> str:
    print("\n" + "=" * 55)
    print(f"  Query: {query}")
    print("=" * 55)

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

    final_state = graph.invoke(initial_state)

    print("\n── Final Answer ──")
    print(final_state["generation"])
    return final_state["generation"]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Agentic RAG — Day 3: LangGraph Graph")
    print("=" * 55)

    print("\nLoading retriever...")
    retriever = HybridRetriever()

    print("\nBuilding graph...")
    graph = build_graph(retriever)

    # Test queries
    # Mix of: specific doc questions, general questions, edge cases
    queries = [
        "What BLEU score did the Transformer achieve on WMT 2014 English-to-German?",
        "What is a neural network?",                        # should go direct
        "How many attention heads does the base model use?",
    ]

    for query in queries:
        run_query(graph, query)


if __name__ == "__main__":
    main()
