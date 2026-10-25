import hashlib

import numpy as np
import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from rag.chunking import chunk_document, split_text
from rag.pipeline import RAGPipeline, build_prompt
from rag.store import Hit, VectorStore


class HashingEmbedding(EmbeddingFunction):
    """Deterministic bag-of-words embedding so tests run offline without model downloads."""

    def __call__(self, input: Documents) -> Embeddings:
        vecs = []
        for text in input:
            v = np.zeros(256, dtype=np.float32)
            for word in text.lower().split():
                v[int(hashlib.md5(word.strip(".,?!").encode()).hexdigest(), 16) % 256] += 1
            n = np.linalg.norm(v)
            vecs.append((v / n if n else v).tolist())
        return vecs


class NoLLM:
    enabled = False


def test_split_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i}." for i in range(300))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0][-20:].split()[-1] in chunks[1]  # overlap carries context forward


def test_split_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=10)


def test_build_prompt_numbers_sources():
    prompt = build_prompt("What?", [Hit("alpha", "a.txt", 0.9), Hit("beta", "b.txt", 0.8)])
    assert "[1] (source: a.txt)" in prompt and "[2] (source: b.txt)" in prompt


def test_retrieval_end_to_end(tmp_path):
    store = VectorStore(path=str(tmp_path), embedding_function=HashingEmbedding())
    store.add(chunk_document("Kubernetes schedules containers onto nodes in a cluster.", "k8s.md"))
    store.add(chunk_document("MLflow tracks experiments, parameters and metrics.", "mlflow.md"))
    rag = RAGPipeline(store, llm=NoLLM(), top_k=2)
    answer = rag.ask("Which tool tracks experiments and metrics?")
    assert answer.mode == "retrieval-only"
    assert answer.sources[0]["source"] == "mlflow.md"
    assert store.sources() == ["k8s.md", "mlflow.md"]
