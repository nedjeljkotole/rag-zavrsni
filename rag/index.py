"""Indeks: matrica vektora + zapisi segmenata (chunk_id) + spremljena konfiguracija."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import List

from .models import Chunk


class VectorIndex:
    def __init__(self, vectorizer, chunks: List[Chunk] | None = None, vectors=None,
                 config: dict | None = None):
        self.vectorizer = vectorizer
        self.chunks: List[Chunk] = chunks or []
        self.vectors = vectors
        self.config = config or {}

    @classmethod
    def build(cls, vectorizer, chunks: List[Chunk], config: dict | None = None) -> "VectorIndex":
        vectors = vectorizer.fit_transform([c.text for c in chunks])
        cfg = dict(config or {})
        cfg.update({"vectorizer": vectorizer.name, "vectorizer_params": vectorizer.params,
                    "dimension": vectorizer.dimension, "metric": "cosine",
                    "num_chunks": len(chunks),
                    "num_documents": len({c.document_id for c in chunks}),
                    "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return cls(vectorizer, chunks, vectors, cfg)

    def store(self, folder: str | Path) -> None:
        """Sprema indeks, zapise segmenata i konfiguraciju (radi ponovljivosti)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "index.pkl", "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "vectors": self.vectors}, f)
        (folder / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "config.json").write_text(json.dumps(self.config, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path) -> "VectorIndex":
        folder = Path(folder)
        with open(folder / "index.pkl", "rb") as f:
            data = pickle.load(f)
        chunks = [Chunk.from_dict(d) for d in json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        return cls(data["vectorizer"], chunks, data["vectors"], config)

    def document_ids(self) -> List[str]:
        return sorted({c.document_id for c in self.chunks})
