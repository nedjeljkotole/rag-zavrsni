"""Sloj reprezentacije. Svi vektorizatori imaju isto sučelje: fit_transform, transform, encode."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfTextVectorizer:
    """Rijetke TF-IDF reprezentacije s unigramima i bigramima (ponovljivi testni način)."""

    name = "tfidf-1-2gram"

    def __init__(self, ngram_range=(1, 2), sublinear_tf: bool = True, min_df: int = 1):
        self.params = {"ngram_range": list(ngram_range), "sublinear_tf": sublinear_tf, "min_df": min_df}
        self._v = TfidfVectorizer(ngram_range=tuple(ngram_range), sublinear_tf=sublinear_tf,
                                  min_df=min_df, lowercase=True, strip_accents=None,
                                  token_pattern=r"(?u)\b\w[\w\-@]*\b")

    def fit_transform(self, texts: Sequence[str]):
        return self._v.fit_transform(texts)

    def transform(self, texts: Sequence[str]):
        # Vektorizator se NE prilagođava ponovno na upitu – prostor značajki ostaje isti.
        return self._v.transform(texts)

    def encode(self, texts: Sequence[str]):
        return self.transform(texts)

    @property
    def dimension(self) -> int:
        return len(self._v.vocabulary_)


class DenseTextVectorizer:
    """Guste vektorske reprezentacije (opcionalno, zahtijeva sentence-transformers)."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
        self.name = model_name
        self.params = {"model": model_name}
        self._m = SentenceTransformer(model_name)

    def fit_transform(self, texts: Sequence[str]):
        return self.encode(texts)

    def transform(self, texts: Sequence[str]):
        return self.encode(texts)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._m.encode(list(texts), normalize_embeddings=True))

    @property
    def dimension(self) -> int:
        return int(self._m.get_sentence_embedding_dimension())


def make_vectorizer(kind: str = "tfidf", **kwargs):
    if kind == "tfidf":
        return TfidfTextVectorizer(**kwargs)
    if kind == "dense":
        return DenseTextVectorizer(**kwargs)
    raise ValueError(f"Nepoznata vrsta vektorizatora: {kind}")
