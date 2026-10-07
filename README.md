# Generiranje vođeno dohvaćanjem (RAG) – referentni prototip

Izvorni kod referentnog prototipa iz završnog rada *Generiranje vođeno dohvaćanjem (RAG)*
(Nedjeljko Tole, Filozofski fakultet Sveučilišta u Mostaru, 2026.).

Prototip odgovara na pitanja nad lokalnom kolekcijom tekstualnih dokumenata i uz svaki
odgovor prikazuje korištene izvore. Sloj za dohvaćanje i generativni sloj su odvojeni, pa se
vektorizator (TF-IDF ili gusti model) i generator (lokalni ekstraktivni ili LLM preko API-ja)
mogu zamijeniti bez promjene ostatka sustava.

## Struktura

```
rag/
  models.py       Document, Chunk, RetrievalResult
  ingestion.py    load_sources, normalize_text, split_and_attach_metadata (segmentacija s preklapanjem, uklanjanje duplikata)
  vectorizers.py  TF-IDF (unigrami i bigrami) i opcionalni gusti model; zajedničko sučelje encode(texts)
  index.py        VectorIndex: matrica vektora + zapisi segmenata (chunk_id) + config.json
  retrieval.py    cosine_similarity, top_k, Retriever (k, prag minimalne sličnosti)
  prompt.py       sustavska uputa (verzionirana) i format_with_sources
  generators.py   ExtractiveGenerator (bez mreže) i OpenAICompatibleGenerator (udaljeni ili lokalni LLM)
  pipeline.py     build_index i RAGPipeline.answer
  cli.py          naredbeni redak
app.py            mrežno sučelje (Flask): pitanje, odgovor, izvori, dijagnostika, ocjena, administracija
evaluation/       kontrolirani test dohvaćanja (15 zapisa, 30 upita) – poglavlje 5.2 rada
data/documents/   primjer kolekcije dokumenata
tests/            jedinični testovi
```

## Instalacija

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Za PDF i DOCX dokumente instalirajte `pypdf` i `python-docx`, a za guste vektorske
reprezentacije `sentence-transformers` (vidi `requirements.txt`).

## Korištenje

Indeksiranje dokumenata iz `data/documents`:

```bash
python -m rag.cli index
```

Pitanje iz naredbenog retka:

```bash
python -m rag.cli ask "Što je generiranje vođeno dohvaćanjem?"
```

Mrežno sučelje (http://127.0.0.1:5000):

```bash
python app.py
```

Administratorska lozinka za dodavanje dokumenata i ponovno indeksiranje zadaje se varijablom
`RAG_ADMIN_TOKEN` (zadano: `admin`).

### Generiranje pomoću jezičnog modela

Zadano se koristi lokalni ekstraktivni generator koji ne zahtijeva mrežu. Za LLM s
OpenAI-kompatibilnim API-jem (npr. OpenAI, Ollama, LM Studio) postavite:

```bash
set RAG_LLM_BASE_URL=http://localhost:11434/v1
set RAG_LLM_MODEL=llama3.1
set RAG_LLM_API_KEY=nebitno
set RAG_GENERATOR=openai
python app.py
```

ili u naredbenom retku `python -m rag.cli ask "..." --generator openai`.

## Evaluacija dohvaćanja

```bash
python -m evaluation.evaluate
```

Rezultati se spremaju u `evaluation/results/summary.json` (Recall@k, MRR i konfiguracija) i
`evaluation/results/per_query.csv` (rang ciljnog zapisa za svaki upit). Uz TF-IDF s unigramima
i bigramima i kosinusnu sličnost dobiveno je:

| Metrika | Rezultat |
|---|---|
| Recall@1 (točnost na prvom mjestu) | 27/30 = 90,0 % |
| Recall@3 | 29/30 = 96,7 % |
| MRR | 0,931 |

Usporedba s gustim reprezentacijama: `python -m evaluation.evaluate dense`.

## Testovi

```bash
pip install pytest
python -m pytest
```
