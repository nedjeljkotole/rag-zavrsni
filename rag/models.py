"""Logičke podatkovne strukture prototipa: Document, Chunk i RetrievalResult."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class Document:
    """Izvorni dokument prije segmentacije."""
    document_id: str
    title: str
    text: str
    source: str = ""
    version: str = "1"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Segment dokumenta koji se indeksira i dohvaća."""
    chunk_id: str
    document_id: str
    position: int          # redni broj segmenta unutar dokumenta
    text: str
    title: str = ""
    source: str = ""
    version: str = "1"
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Chunk":
        return Chunk(**d)


@dataclass
class RetrievalResult:
    """Rezultat dohvaćanja: segment, ocjena sličnosti i rang."""
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "score": round(self.score, 4), **self.chunk.to_dict()}
