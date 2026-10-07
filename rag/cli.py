"""Naredbeni redak prototipa.

    python -m rag.cli index  [--docs data/documents] [--index data/index]
    python -m rag.cli ask    "Pitanje?" [--k 3] [--generator extractive|openai]
"""
from __future__ import annotations

import argparse
import json

from .generators import make_generator
from .index import VectorIndex
from .pipeline import RAGPipeline, build_index


def main() -> None:
    ap = argparse.ArgumentParser(prog="rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index", help="indeksiraj dokumente")
    pi.add_argument("--docs", default="data/documents")
    pi.add_argument("--index", default="data/index")
    pi.add_argument("--vectorizer", default="tfidf", choices=["tfidf", "dense"])
    pi.add_argument("--max-words", type=int, default=200)
    pi.add_argument("--overlap", type=int, default=30)
    pa = sub.add_parser("ask", help="postavi pitanje")
    pa.add_argument("question")
    pa.add_argument("--index", default="data/index")
    pa.add_argument("--k", type=int, default=3)
    pa.add_argument("--min-score", type=float, default=0.05)
    pa.add_argument("--generator", default="extractive", choices=["extractive", "openai"])
    pa.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.cmd == "index":
        idx = build_index(args.docs, args.index, args.vectorizer, args.max_words, args.overlap)
        print(f"Indeksirano {idx.config['num_documents']} dokumenata u {idx.config['num_chunks']} segmenata.")
        return

    pipe = RAGPipeline(VectorIndex.load(args.index), make_generator(args.generator),
                       k=args.k, min_score=args.min_score)
    out = pipe.answer(args.question)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    print(f"\nOdgovor:\n{out['answer']}\n\nIzvori:")
    for s in out["sources"]:
        print(f"  [{s['rank']}] {s['title']} ({s['source']}, segment {s['position']}) – sličnost {s['score']}")
    print(f"\nDohvaćanje: {out['diagnostics']['retrieval_ms']} ms, generiranje: {out['diagnostics']['generation_ms']} ms")


if __name__ == "__main__":
    main()
