"""Kontrolirani test podsustava za dohvaćanje (poglavlje 5.2 rada).

Pokretanje (iz korijenske mape projekta):
    python -m evaluation.evaluate

Ulaz:  evaluation/corpus/*.txt   – 15 tematskih zapisa
       evaluation/questions.json – 30 upita s označenim ciljnim zapisom
Izlaz: evaluation/results/summary.json   – Recall@1, Recall@3, MRR i konfiguracija
       evaluation/results/per_query.csv  – rang ciljnog zapisa za svaki upit
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.index import VectorIndex  # noqa: E402
from rag.ingestion import load_sources, split_and_attach_metadata  # noqa: E402
from rag.retrieval import Retriever  # noqa: E402
from rag.vectorizers import make_vectorizer  # noqa: E402

HERE = Path(__file__).resolve().parent
KS = (1, 2, 3, 5)


def target_rank(ranked, target: str) -> int | None:
    """Rang prvog segmenta koji pripada ciljnom zapisu."""
    for r in ranked:
        if r.chunk.document_id == target:
            return r.rank
    return None


def main(vectorizer_kind: str = "tfidf") -> dict:
    documents = load_sources(HERE / "corpus")
    chunks = split_and_attach_metadata(documents, max_words=200, overlap_words=30)
    index = VectorIndex.build(make_vectorizer(vectorizer_kind), chunks)
    retriever = Retriever(index, k=3, min_score=0.0)
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))

    rows = []
    for q in questions:
        ranked = retriever.rank_all(q["question"])
        rank = target_rank(ranked, q["target"])
        top1 = ranked[0].chunk.document_id
        rows.append({"id": q["id"], "question": q["question"], "target": q["target"],
                     "rank": rank, "top1": top1, "top1_score": round(ranked[0].score, 4)})

    n = len(rows)
    recall = {f"recall@{k}": sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n for k in KS}
    mrr = sum(1.0 / r["rank"] for r in rows if r["rank"]) / n
    summary = {
        "num_documents": len(documents), "num_chunks": len(chunks), "num_queries": n,
        "hits@1": sum(1 for r in rows if r["rank"] == 1),
        "hits@3": sum(1 for r in rows if r["rank"] and r["rank"] <= 3),
        **{k: round(v, 4) for k, v in recall.items()}, "mrr": round(mrr, 4),
        "config": index.config,
        "errors_top1": [r for r in rows if r["rank"] != 1],
    }
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Zapisa: {len(documents)}, segmenata: {len(chunks)}, upita: {n}")
    for k in KS:
        print(f"Recall@{k}: {recall[f'recall@{k}']*100:.1f} %")
    print(f"MRR: {mrr:.3f}")
    print("Upiti kod kojih ciljni zapis nije na prvom mjestu:")
    for r in summary["errors_top1"]:
        print(f"  {r['id']} (rang {r['rank']}): {r['question']}  -> prvi: {r['top1']}")
    return summary


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tfidf")
