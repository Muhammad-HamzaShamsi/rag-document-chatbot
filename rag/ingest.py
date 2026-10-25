"""CLI: index a folder of documents.

    python -m rag.ingest data/sample
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .chunking import chunk_document, load_text
from .store import VectorStore

SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".rst"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "800")))
    parser.add_argument("--overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "150")))
    args = parser.parse_args()

    store = VectorStore(path=os.getenv("CHROMA_DIR", "chroma_db"))
    for path in sorted(args.folder.rglob("*")):
        if path.suffix.lower() in SUPPORTED:
            chunks = chunk_document(load_text(path), path.name, args.chunk_size, args.overlap)
            print(f"{path.name}: {store.add(chunks)} chunks")
    print(f"Total chunks in store: {store.count()}")


if __name__ == "__main__":
    main()
