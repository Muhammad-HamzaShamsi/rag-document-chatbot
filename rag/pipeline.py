"""Retrieval-augmented generation: retrieve chunks, build a grounded prompt, call the LLM."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .store import Hit, VectorStore

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers strictly from the provided context. "
    "Cite sources in square brackets like [1], [2]. If the answer is not in the context, "
    "say you don't know instead of guessing."
)


@dataclass
class Answer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    mode: str = "llm"


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[{i}] (source: {h.source})\n{h.text}" for i, h in enumerate(hits, 1))
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


class LLM:
    """Thin wrapper around any OpenAI-compatible chat endpoint (OpenAI, Groq, Ollama...)."""

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def complete(self, prompt: str) -> str:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()


class RAGPipeline:
    def __init__(self, store: VectorStore, llm: LLM | None = None, top_k: int | None = None):
        self.store = store
        self.llm = llm or LLM()
        self.top_k = top_k or int(os.getenv("TOP_K", "4"))

    def ask(self, question: str) -> Answer:
        hits = self.store.search(question, k=self.top_k)
        sources = [{"id": i, "source": h.source, "score": h.score, "text": h.text[:300]}
                   for i, h in enumerate(hits, 1)]
        if not hits:
            return Answer("No documents have been ingested yet.", [], mode="empty")
        if not self.llm.enabled:
            # Retrieval-only fallback: useful for demos and tests without an API key.
            best = hits[0]
            return Answer(f"Most relevant passage [1] from {best.source}:\n\n{best.text}",
                          sources, mode="retrieval-only")
        return Answer(self.llm.complete(build_prompt(question, hits)), sources, mode="llm")
