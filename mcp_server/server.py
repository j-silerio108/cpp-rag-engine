"""MCP server exposing semantic search over a local notes corpus.

This process does not reimplement the search -- it embeds the query with the
same local model used to build the index, then hands the vector to the
compiled C++ engine (rag_cli.exe) over a subprocess call and returns its
--json output. The engine is still what ranks results; this file only gives
it a new way to be asked a question (stdio/MCP instead of argv).
"""

import json
import os
import subprocess
from pathlib import Path

from fastembed import TextEmbedding
from mcp.server.mcpserver import MCPServer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
REPO_ROOT = Path(__file__).resolve().parent.parent
RAG_CLI = REPO_ROOT / "build" / "rag_cli.exe"
DEFAULT_CSV = Path(os.environ.get("RAGENGINE_CSV_PATH", str(REPO_ROOT / "data" / "documents.csv")))

mcp = MCPServer("ragengine")
_model = TextEmbedding(model_name=MODEL_NAME)


@mcp.tool()
def search_notes(query: str, top_k: int = 3) -> list[dict]:
    """Semantically search the local notes corpus and return the closest matches.

    Matches by meaning, not keyword overlap: a query about "pets" can surface
    a note that only ever says "my cat Whiskers". Returns up to top_k results
    as {id, score, text}, most similar first.
    """
    if not RAG_CLI.exists():
        raise RuntimeError(f"rag_cli.exe not found at {RAG_CLI} -- build it first with `cmake --build build`")
    if not DEFAULT_CSV.exists():
        raise RuntimeError(f"No document index at {DEFAULT_CSV} -- run tools/embed_documents.py first")

    vector = next(_model.embed([query]))
    vector_arg = "|".join(f"{v:.6f}" for v in vector)

    result = subprocess.run(
        [str(RAG_CLI), str(DEFAULT_CSV), vector_arg, str(top_k), "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rag_cli.exe failed: {result.stderr.strip()}")

    return json.loads(result.stdout)


if __name__ == "__main__":
    mcp.run()
