import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph.state import CompiledStateGraph
from hybrid_retriever import HybridRetriever
from graph import build_graph, run_query

# ─────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════╗
║         Agentic Document Q&A System              ║
║         Powered by LangGraph + Groq              ║
╚══════════════════════════════════════════════════╝

Commands:
  Type any question → get an answer
  'quit' or 'exit'  → exit
  'clear'           → clear screen
  'help'            → show this menu
"""

# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

def check_env():
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set in .env file")
        print("Create a .env file with: GROQ_API_KEY=your_key_here")
        sys.exit(1)

    chroma_dir = "./chroma_db"
    if not os.path.exists(chroma_dir):
        print("ERROR: ChromaDB not found. Run ingest.py first.")
        print("  python ingest.py")
        sys.exit(1)

def load_system() -> CompiledStateGraph:
    print("Starting up...")
    print("  Loading retriever (this takes ~10 seconds first time)...")
    retriever = HybridRetriever()
    print("  Building graph...")
    graph = build_graph(retriever)
    print("  Ready.\n")
    return graph

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    check_env()
    print(BANNER)

    graph = load_system()

    print("Ask anything about your documents. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        if query.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)
            continue

        if query.lower() == "help":
            print(BANNER)
            continue

        # Run the agentic graph
        print()
        answer = run_query(graph, query)
        print(f"\nAnswer: {answer}\n")
        print("-" * 55)

if __name__ == "__main__":
    main()