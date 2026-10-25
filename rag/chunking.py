"""Document loading and chunking utilities (no external services required)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    index: int


def load_text(path: Path) -> str:
    """Read .txt / .md / .pdf files into plain text."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md", ".markdown", ".rst"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix}")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into ~chunk_size character windows, preferring paragraph/sentence breaks.

    Consecutive chunks share `overlap` characters so answers spanning a boundary
    are still retrievable.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = normalize(text)
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            # Prefer to cut at a paragraph, then a sentence, then a space.
            for sep in ("\n\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > chunk_size * 0.5:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(
    text: str, source: str, chunk_size: int = 800, overlap: int = 150
) -> list[Chunk]:
    out = []
    for i, piece in enumerate(split_text(text, chunk_size, overlap)):
        digest = hashlib.sha1(f"{source}:{i}:{piece}".encode()).hexdigest()[:16]
        out.append(Chunk(id=digest, text=piece, source=source, index=i))
    return out
