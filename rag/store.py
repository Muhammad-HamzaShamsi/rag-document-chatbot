"""Vector store backed by ChromaDB (local, persistent)."""
from __future__ import annotations

from dataclasses import dataclass

import chromadb

from .chunking import Chunk


@dataclass
class Hit:
    text: str
    source: str
    score: float


class VectorStore:
    def __init__(
        self, path: str = "chroma_db", collection: str = "documents", embedding_function=None
    ):
        """`embedding_function=None` uses Chroma's built-in all-MiniLM-L6-v2 (ONNX, runs on CPU)."""
        self.client = chromadb.PersistentClient(path=path)
        kwargs = {"metadata": {"hnsw:space": "cosine"}}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self._kwargs = kwargs
        self.collection = self.client.get_or_create_collection(collection, **kwargs)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "index": c.index} for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if self.count() == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(k, self.count()))
        rows = zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True)
        return [
            Hit(text=doc, source=meta.get("source", "?"), score=round(1 - dist, 4))
            for doc, meta, dist in rows
        ]

    def sources(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"] or []
        return sorted({m.get("source", "?") for m in metas})

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name, **self._kwargs)
