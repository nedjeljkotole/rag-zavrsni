import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.index import VectorIndex
from rag.ingestion import Document, split_and_attach_metadata, split_text
from rag.pipeline import RAGPipeline
from rag.retrieval import Retriever, top_k
from rag.vectorizers import make_vectorizer
import numpy as np


def _index():
    docs = [Document("a", "A", "Transformer koristi mehanizam samopažnje."),
            Document("b", "B", "Chroma je vektorska baza s kolekcijama i metapodacima."),
            Document("c", "C", "BM25 je algoritam pretraživanja po ključnim riječima.")]
    return VectorIndex.build(make_vectorizer("tfidf"), split_and_attach_metadata(docs, min_words=1))


def test_split_overlap():
    text = " ".join(f"r{i}." for i in range(500))
    parts = split_text(text, max_words=100, overlap_words=20)
    assert len(parts) > 1
    assert all(len(p.split()) <= 120 for p in parts)


def test_duplicates_removed():
    docs = [Document("a", "A", "Isti tekst segmenta ovdje."), Document("b", "B", "Isti tekst segmenta ovdje.")]
    assert len(split_and_attach_metadata(docs)) == 1


def test_top_k_order():
    assert top_k(np.array([0.1, 0.9, 0.5]), 2) == [1, 2]


def test_retrieval_finds_target():
    r = Retriever(_index(), k=1).retrieve("Što je Chroma?")
    assert r and r[0].chunk.document_id == "b"


def test_no_context_message():
    out = RAGPipeline(_index(), min_score=0.99).answer("Nešto potpuno nevezano")
    assert "nije pronađeno" in out["answer"]
