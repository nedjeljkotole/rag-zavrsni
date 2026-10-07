"""Sustavska uputa (verzionirana) i oblikovanje dohvaćenog konteksta."""
from __future__ import annotations

from typing import List

from .models import RetrievalResult

PROMPT_VERSION = "1.0"

SYSTEM_PROMPT = (
    "Odgovori na pitanje koristeći isključivo dostavljeni kontekst. "
    "Ako kontekst ne sadrži dovoljno informacija, to izričito navedi. "
    "Nemoj izmišljati izvore. Uz tvrdnje navedi oznaku izvora u obliku [Izvor N]. "
    "Odgovaraj na hrvatskom jeziku, sažeto i jasno."
)

NO_CONTEXT_MESSAGE = "U indeksiranim dokumentima nije pronađeno dovoljno informacija za odgovor na ovo pitanje."


def format_with_sources(results: List[RetrievalResult]) -> str:
    """Oblikuje rezultate u blokove „Izvor 1“, „Izvor 2“ … s nazivom dokumenta."""
    blocks = []
    for r in results:
        c = r.chunk
        blocks.append(f"[Izvor {r.rank}] {c.title} ({c.source}, segment {c.position})\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, context: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Kontekst:\n{context}\n\nPitanje: {question}"},
    ]
