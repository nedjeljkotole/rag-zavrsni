"""Mrežno sučelje prototipa (Flask).

    python app.py            -> http://127.0.0.1:5000

Korisnički dio: pitanje, odgovor i korišteni izvori.
Dijagnostika: prvih k segmenata, ocjene sličnosti i trajanje koraka.
Administracija: dodavanje dokumenta i ponovno indeksiranje (zaštićeno lozinkom RAG_ADMIN_TOKEN).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from rag.generators import make_generator
from rag.index import VectorIndex
from rag.ingestion import SUPPORTED
from rag.pipeline import RAGPipeline, build_index

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "data" / "documents"
INDEX = ROOT / "data" / "index"
FEEDBACK = ROOT / "data" / "feedback.jsonl"
ADMIN_TOKEN = os.environ.get("RAG_ADMIN_TOKEN", "admin")
GENERATOR = os.environ.get("RAG_GENERATOR", "extractive")

app = Flask(__name__)
_pipeline: RAGPipeline | None = None


def pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        if not (INDEX / "index.pkl").exists():
            build_index(DOCS, INDEX)
        _pipeline = RAGPipeline(VectorIndex.load(INDEX), make_generator(GENERATOR), k=3, min_score=0.05)
    return _pipeline


@app.route("/", methods=["GET", "POST"])
def home():
    result, error = None, None
    if request.method == "POST":
        q = request.form.get("question", "").strip()[:1000]
        if not q:
            error = "Upišite pitanje."
        else:
            try:
                result = pipeline().answer(q)
            except Exception as exc:  # prikaz pogreške u sučelju
                error = f"Pogreška: {exc}"
    p = pipeline()
    return render_template("index.html", result=result, error=error, config=p.index.config,
                           documents=p.index.document_ids(), message=request.args.get("msg"))


@app.post("/feedback")
def feedback():
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "question": request.form.get("question", ""),
             "answer": request.form.get("answer", ""), "rating": request.form.get("rating", ""),
             "comment": request.form.get("comment", "")}
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return redirect(url_for("home", msg="Hvala na povratnoj informaciji."))


@app.post("/admin/upload")
def upload():
    global _pipeline
    if request.form.get("token") != ADMIN_TOKEN:
        return redirect(url_for("home", msg="Neispravna administratorska lozinka."))
    f = request.files.get("file")
    if not f or not f.filename:
        return redirect(url_for("home", msg="Odaberite datoteku."))
    name = secure_filename(f.filename)
    if Path(name).suffix.lower() not in SUPPORTED:
        return redirect(url_for("home", msg="Nepodržan format datoteke."))
    DOCS.mkdir(parents=True, exist_ok=True)
    f.save(DOCS / name)
    build_index(DOCS, INDEX)
    _pipeline = None
    return redirect(url_for("home", msg=f"Dokument {name} dodan i indeks ponovno izgrađen."))


@app.post("/admin/reindex")
def reindex():
    global _pipeline
    if request.form.get("token") != ADMIN_TOKEN:
        return redirect(url_for("home", msg="Neispravna administratorska lozinka."))
    build_index(DOCS, INDEX)
    _pipeline = None
    return redirect(url_for("home", msg="Indeks je ponovno izgrađen."))


if __name__ == "__main__":
    app.run(debug=False)
