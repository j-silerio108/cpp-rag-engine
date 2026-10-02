"""Turns a folder of .txt notes into the CSV format ragengine's C++ loader expects:
    id,"text",v1|v2|v3|...

Each .txt file is one document. Embeddings come from a small local ONNX model
(BAAI/bge-small-en-v1.5, ~130MB) run via fastembed -- no network calls at
embedding time, only the one-time model download on first use.

Usage:
    tools/venv/Scripts/python.exe tools/embed_documents.py \
        --input-dir data/notes --output data/documents.csv
"""

import argparse
import csv
from pathlib import Path

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_notes(input_dir: Path) -> list[tuple[int, str]]:
    paths = sorted(input_dir.glob("*.txt"))
    if not paths:
        raise SystemExit(f"No .txt files found in {input_dir}")
    # Collapse to one line per document: the C++ CSV reader is line-based and
    # doesn't handle embedded newlines inside quoted fields.
    return [
        (doc_id, " ".join(path.read_text(encoding="utf-8").split()))
        for doc_id, path in enumerate(paths, start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/notes"))
    parser.add_argument("--output", type=Path, default=Path("data/documents.csv"))
    args = parser.parse_args()

    notes = load_notes(args.input_dir)
    print(f"Embedding {len(notes)} documents with {MODEL_NAME}...")

    model = TextEmbedding(model_name=MODEL_NAME)
    embeddings = list(model.embed([text for _, text in notes]))

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        for (doc_id, text), vector in zip(notes, embeddings):
            embedding_field = "|".join(f"{v:.6f}" for v in vector)
            writer.writerow([doc_id, text, embedding_field])

    print(f"Wrote {len(notes)} documents ({len(embeddings[0])}-dim embeddings) to {args.output}")


if __name__ == "__main__":
    main()
