"""Embeds a single query string with the same model used for documents, and
prints the pipe-separated vector that rag_cli.exe expects as its query argument.

Usage:
    tools/venv/Scripts/python.exe tools/embed_query.py "what should I do with my dog"
"""

import sys

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} \"query text\"")

    model = TextEmbedding(model_name=MODEL_NAME)
    vector = next(model.embed([sys.argv[1]]))
    print("|".join(f"{v:.6f}" for v in vector))


if __name__ == "__main__":
    main()
