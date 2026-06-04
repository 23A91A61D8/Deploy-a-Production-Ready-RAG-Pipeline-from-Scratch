"""
main.py
-------
Entry point for the RAG Pipeline application.

Modes:
  --ingest          Run the ingestion phase
  --query "..."     Run a single query and exit
  --interactive     Launch an interactive Q&A session (default)
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _print_result(result: dict, show_prompt: bool = False) -> None:
    """Pretty-print query results to stdout."""
    print("\n" + "=" * 70)
    print(f"❓ QUESTION:\n   {result['query']}")
    print("-" * 70)
    print(f"🤖 ANSWER:\n   {result['answer']}")
    print("-" * 70)
    print("📚 SOURCES:")
    if result["sources"]:
        for i, src in enumerate(result["sources"], start=1):
            doc_name = os.path.basename(src.get("source", "unknown"))
            chunk_idx = src.get("chunk_index", "?")
            score = src.get("score", 0.0)
            print(f"\n  [{i}] {doc_name}  (chunk #{chunk_idx}, similarity={score:.4f})")
            # Show a short excerpt of the source chunk
            excerpt = src["text"][:300].replace("\n", " ")
            if len(src["text"]) > 300:
                excerpt += "…"
            print(f"      \"{excerpt}\"")
    else:
        print("  No sources retrieved.")

    if show_prompt:
        print("-" * 70)
        print("📝 PROMPT SENT TO LLM:")
        print(result["prompt"])

    print("=" * 70 + "\n")


def run_ingest(args: argparse.Namespace) -> None:
    """Run the ingestion phase."""
    from src.ingest import ingest_documents

    stats = ingest_documents(
        docs_dir=args.docs_dir,
        chunking_strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print("\n✅ Ingestion complete:")
    print(f"   Documents loaded : {stats['documents_loaded']}")
    print(f"   Chunks created   : {stats['chunks_created']}")
    print(f"   Chunks stored    : {stats['chunks_stored']}")


def run_query(query: str, show_prompt: bool = False) -> None:
    """Run a single query."""
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    result = pipeline.query(query)
    _print_result(result, show_prompt=show_prompt)


def run_interactive(show_prompt: bool = False) -> None:
    """Interactive Q&A loop."""
    from src.rag_pipeline import RAGPipeline

    print("\n" + "=" * 70)
    print("  🔍 RAG Pipeline — Interactive Mode")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 70 + "\n")

    pipeline = RAGPipeline()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            result = pipeline.query(user_input)
            _print_result(result, show_prompt=show_prompt)
        except Exception as exc:
            logger.error("Query failed: %s", exc, exc_info=True)
            print(f"\n⚠️  Error: {exc}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production-Ready RAG Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--ingest",
        action="store_true",
        help="Run the document ingestion phase.",
    )
    mode_group.add_argument(
        "--query",
        type=str,
        metavar="QUESTION",
        help="Run a single query and exit.",
    )
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Launch interactive Q&A session (default if no mode specified).",
    )

    # Ingestion options
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=os.getenv("DOCS_DIR", "./data/documents"),
        help="Directory containing source documents for ingestion.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=os.getenv("CHUNKING_STRATEGY", "sentence_aware"),
        choices=["fixed_size", "sentence_aware", "sliding_window"],
        help="Chunking strategy.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", 400)),
        help="Target chunk size in tokens.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", 80)),
        help="Overlap tokens between chunks.",
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Print the full LLM prompt alongside the answer (debug mode).",
    )

    args = parser.parse_args()

    if args.ingest:
        run_ingest(args)
    elif args.query:
        run_query(args.query, show_prompt=args.show_prompt)
    else:
        # Default: interactive
        run_interactive(show_prompt=args.show_prompt)


if __name__ == "__main__":
    main()
