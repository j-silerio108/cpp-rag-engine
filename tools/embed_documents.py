"""Turns a set of .txt/.md files into the CSV format ragengine's C++ loader
expects:
    id,"text",v1|v2|v3|...

Each file is split into paragraph-sized chunks and each chunk becomes one
document/embedding -- not one embedding per whole file -- so a long file's
embedding isn't an average over everything in it, and a search result is a
short, relevant passage instead of the entire file. Embeddings come from a
small local ONNX model (BAAI/bge-small-en-v1.5, ~130MB) run via fastembed --
no network calls at embedding time, only the one-time model download on
first use.

Sources can be explicit directories (--input-dir, repeatable) and/or glob
patterns matching directories (--input-glob, repeatable, e.g. a pattern
that expands to every project's memory folder). Each source directory is
scanned (non-recursively) for *.txt and *.md files; a file named exactly
"MEMORY.md" is skipped, since in a Claude Code memory folder that's just an
index of links to the other files, not content worth searching.

Chunking: paragraphs are split on blank lines (so a markdown table or list
stays intact as one chunk unless it's unusually long), tagged with the
nearest preceding "## heading" if there is one, and any paragraph longer
than --max-chunk-words is further split into fixed-size word windows. For a
file inside a directory named "memory", every chunk is prefixed with its
source project and filename (and heading, if any) -- e.g.
"[carMarket] carmarket-architecture > Database schema: ..." -- since
otherwise there's no way to tell which project, file, or section a search
result came from. Any YAML frontmatter (a leading "---\n...\n---" block) is
stripped before chunking -- it's structural metadata, not prose.

Usage:
    tools/venv/Scripts/python.exe tools/embed_documents.py \
        --input-dir data/notes \
        --input-glob "C:/Users/joses/.claude/projects/*/memory" \
        --output data/documents.csv
"""

import argparse
import csv
import glob
import re
from pathlib import Path

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_MAX_CHUNK_WORDS = 150

FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.S)

# Claude Code encodes a project's working directory into its storage folder
# name (e.g. C:\Users\joses\OneDrive\Documents\carMarket ->
# C--Users-joses-OneDrive-Documents-carMarket); these strip the common
# prefixes down to just the project name for a readable label.
PROJECT_LABEL_PREFIXES = [
    re.compile(r"^C--Users-joses-OneDrive-Documents-"),
    re.compile(r"^C--Users-joses-repos-"),
    re.compile(r"^C--Users-joses-"),
    re.compile(r"^C--dev-"),
]


def project_label(memory_dir: Path) -> str:
    label = memory_dir.parent.name
    for pattern in PROJECT_LABEL_PREFIXES:
        if pattern.match(label):
            return pattern.sub("", label)
    return label


def split_into_chunks(text: str, max_chunk_words: int) -> list[tuple[str, str]]:
    """Splits text into (heading, chunk_text) pairs. heading is the nearest
    preceding '#'-style markdown heading seen so far, or '' if none yet."""
    paragraphs: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            paragraph = "\n".join(current_lines).strip()
            if paragraph:
                paragraphs.append((current_heading, paragraph))
            current_lines.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            current_heading = stripped.lstrip("#").strip()
        elif stripped == "":
            flush()
        else:
            current_lines.append(line)
    flush()

    chunks: list[tuple[str, str]] = []
    for heading, paragraph in paragraphs:
        words = paragraph.split()
        if len(words) <= max_chunk_words:
            chunks.append((heading, paragraph))
        else:
            for i in range(0, len(words), max_chunk_words):
                chunks.append((heading, " ".join(words[i : i + max_chunk_words])))
    return chunks


def collect_paths(input_dirs: list[Path], input_globs: list[str]) -> list[Path]:
    candidate_dirs = list(input_dirs)
    for pattern in input_globs:
        candidate_dirs.extend(Path(p) for p in sorted(glob.glob(pattern)) if Path(p).is_dir())

    seen_dirs = set()
    paths = []
    for d in candidate_dirs:
        resolved = d.resolve()
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        for file_path in sorted(d.glob("*.txt")) + sorted(d.glob("*.md")):
            if file_path.name.lower() == "memory.md":
                continue
            paths.append(file_path)
    return paths


def load_chunks(paths: list[Path], max_chunk_words: int) -> list[tuple[int, str]]:
    if not paths:
        raise SystemExit("No .txt/.md files found in any source")

    notes = []
    next_id = 1
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".md":
            raw = FRONTMATTER_RE.sub("", raw, count=1)

        is_memory_file = path.parent.name == "memory"
        label = project_label(path.parent) if is_memory_file else None

        for heading, chunk in split_into_chunks(raw, max_chunk_words):
            collapsed = " ".join(chunk.split())
            if not collapsed:
                continue
            if is_memory_file:
                location = f"{path.stem} > {heading}" if heading else path.stem
                text = f"[{label}] {location}: {collapsed}"
            else:
                text = collapsed
            notes.append((next_id, text))
            next_id += 1
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, action="append", default=[])
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--max-chunk-words", type=int, default=DEFAULT_MAX_CHUNK_WORDS)
    parser.add_argument("--output", type=Path, default=Path("data/documents.csv"))
    args = parser.parse_args()

    input_dirs = args.input_dir or ([Path("data/notes")] if not args.input_glob else [])
    paths = collect_paths(input_dirs, args.input_glob)
    notes = load_chunks(paths, args.max_chunk_words)
    print(f"Embedding {len(notes)} chunks from {len(paths)} files with {MODEL_NAME}...")

    model = TextEmbedding(model_name=MODEL_NAME)
    embeddings = list(model.embed([text for _, text in notes]))

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        for (doc_id, text), vector in zip(notes, embeddings):
            embedding_field = "|".join(f"{v:.6f}" for v in vector)
            writer.writerow([doc_id, text, embedding_field])

    print(f"Wrote {len(notes)} chunks ({len(embeddings[0])}-dim embeddings) to {args.output}")


if __name__ == "__main__":
    main()
