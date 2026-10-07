"""Povezivanje slojeva: indeksiranje i odgovaranje na upit (vidi Prilog 1 u radu)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

from .generators import BaseGenerator, ExtractiveGenerator
from .index import VectorIndex
from .ingestion import load_sources, split_and_attach_metadata
from .prompt import PROMPT_VERSION, format_with_sources
from .retrieval import Retriever
from .vectorizers import make_vectorizer


def build_index(documents_dir: str | Path, index_dir: str | Path, vectorizer_kind: str = "tfidf",
                max_words: int = 200, overlap_words: int = 30) -> VectorIndex:
    """INDEKSIRANJE: load_sources -> split_and_attach_metadata -> fit_transform -> store."""
    documents = load_sources(documents_dir)
    chunks = split_and_attach_metadata(documents, max_words=max_words, overlap_words=overlap_words)
    if not chunks:
        raise ValueError(f"U mapi {documents_dir} nema dokumenata za indeksiranje.")
    vectorizer = make_vectorizer(vectorizer_kind)
    index = VectorIndex.build(vectorizer, chunks, config={
        "documents_dir": str(documents_dir), "chunking": {"max_words": max_words,
                                                          "overlap_words": overlap_words}})
    index.store(index_dir)
    return index


class RAGPipeline:
    def __init__(self, index: VectorIndex, generator: BaseGenerator | None = None,
                 k: int = 3, min_score: float = 0.05):
        self.index = index
        self.retriever = Retriever(index, k=k, min_score=min_score)
        self.generator = generator or ExtractiveGenerator()

    def answer(self, question: str) -> Dict:
        """UPIT: transform -> cosine_similarity -> top_k -> format_with_sources -> generator."""
        t0 = time.perf_counter()
        results = self.retriever.retrieve(question)
        t1 = time.perf_counter()
        answer = self.generator.generate(question, results)
        t2 = time.perf_counter()
        return {
            "question": question,
            "answer": answer,
            "sources": [r.to_dict() for r in results],
            "context": format_with_sources(results),
            "diagnostics": {"retrieval_ms": round((t1 - t0) * 1000, 2),
                            "generation_ms": round((t2 - t1) * 1000, 2),
                            "k": self.retriever.k, "min_score": self.retriever.min_score,
                            "generator": self.generator.name, "prompt_version": PROMPT_VERSION},
        }
