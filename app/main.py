"""FastAPI app: upload documents, ask questions, get grounded answers with citations."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.chunking import chunk_document, load_text
from rag.pipeline import RAGPipeline
from rag.store import VectorStore

load_dotenv()

STATIC = Path(__file__).parent / "static"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
ALLOWED = {".pdf", ".txt", ".md", ".markdown", ".rst"}

app = FastAPI(title="RAG Document Chatbot", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_state: dict = {}


def get_pipeline() -> RAGPipeline:
    if "pipeline" not in _state:
        store = VectorStore(path=os.getenv("CHROMA_DIR", "chroma_db"))
        _state["pipeline"] = RAGPipeline(store)
    return _state["pipeline"]


class Question(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    p = get_pipeline()
    return {"status": "ok", "chunks": p.store.count(), "llm_enabled": p.llm.enabled}


@app.get("/documents")
def documents():
    p = get_pipeline()
    return {"documents": p.store.sources(), "chunks": p.store.count()}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    p = get_pipeline()
    report = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in ALLOWED:
            raise HTTPException(415, f"{f.filename}: unsupported type {suffix}")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await f.read())
            tmp_path = Path(tmp.name)
        try:
            chunks = chunk_document(load_text(tmp_path), f.filename, CHUNK_SIZE, CHUNK_OVERLAP)
        finally:
            tmp_path.unlink(missing_ok=True)
        report.append({"file": f.filename, "chunks": p.store.add(chunks)})
    return {"ingested": report, "total_chunks": p.store.count()}


@app.post("/ask")
def ask(q: Question):
    a = get_pipeline().ask(q.question)
    return {"answer": a.answer, "sources": a.sources, "mode": a.mode}


@app.delete("/documents")
def reset():
    get_pipeline().store.reset()
    return {"status": "cleared"}
