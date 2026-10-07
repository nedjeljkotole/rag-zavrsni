"""Sloj za dohvaćanje: kosinusna sličnost, odabir k najboljih i prag minimalne sličnosti."""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as _cos

from .models import Chunk, RetrievalResult


def cosine_similarity(query_vector, vectors) -> np.ndarray:
    """cos(q, d) = (q · d) / (||q|| ||d||) za jedan upit i sve vektore indeksa."""
    return _cos(query_vector, vectors).ravel()


def top_k(scores: np.ndarray, k: int = 3) -> List[int]:
    """Indeksi k najvećih vrijednosti, silazno (stabilno za jednake vrijednosti)."""
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order[:k]]


class Retriever:
    def __init__(self, index, k: int = 3, min_score: float = 0.05):
        self.index = index
        self.k = k
        self.min_score = min_score

    def retrieve(self, question: str, k: int | None = None) -> List[RetrievalResult]:
        question = (question or "").strip()
        if not question:
            raise ValueError("Upit je prazan.")
        k = k or self.k
        query_vector = self.index.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.index.vectors)
        results: List[RetrievalResult] = []
        for rank, i in enumerate(top_k(scores, k), start=1):
            if scores[i] < self.min_score:
                break
            chunk: Chunk = self.index.chunks[i]
            results.append(RetrievalResult(chunk=chunk, score=float(scores[i]), rank=rank))
        return results

    def rank_all(self, question: str) -> List[RetrievalResult]:
        """Svi segmenti rangirani po sličnosti (koristi se u evaluaciji za MRR)."""
        query_vector = self.index.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.index.vectors)
        return [RetrievalResult(chunk=self.index.chunks[i], score=float(scores[i]), rank=r)
                for r, i in enumerate(top_k(scores, len(scores)), start=1)]
