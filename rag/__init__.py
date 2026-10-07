"""Referentni prototip sustava za generiranje vođeno dohvaćanjem (RAG).

Paket je organiziran u neovisne slojeve:
- ingestion   – učitavanje dokumenata, normalizacija i segmentacija
- vectorizers – izrada vektorskih reprezentacija (TF-IDF ili gusti model)
- index       – spremanje vektora i zapisa segmenata
- retrieval   – kosinusna sličnost i odabir k najboljih segmenata
- prompt      – sustavska uputa i oblikovanje konteksta
- generators  – generativni adapteri (ekstraktivni, OpenAI-kompatibilni API)
- pipeline    – povezivanje svih slojeva u jedan tok
"""

__version__ = "1.0.0"
