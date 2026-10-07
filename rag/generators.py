"""Generativni sloj definiran kao adapter: prima pitanje i kontekst, vraća tekst odgovora."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import List

from .models import RetrievalResult
from .prompt import NO_CONTEXT_MESSAGE, build_messages, format_with_sources


class BaseGenerator:
    name = "base"

    def generate(self, question: str, results: List[RetrievalResult]) -> str:
        raise NotImplementedError


class ExtractiveGenerator(BaseGenerator):
    """Lokalni generator bez jezičnog modela.

    Iz dohvaćenih segmenata bira rečenice koje dijele najviše riječi s pitanjem i
    vraća ih uz oznake izvora. Služi za rad bez mreže i za demonstraciju toka sustava.
    """
    name = "extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, question: str, results: List[RetrievalResult]) -> str:
        if not results:
            return NO_CONTEXT_MESSAGE
        q = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 3}
        scored = []
        for r in results:
            for s in re.split(r"(?<=[.!?])\s+|\n+", r.chunk.text):
                if len(s.split()) < 5:  # preskače naslove i prekratke fragmente
                    continue
                words = set(re.findall(r"\w+", s.lower()))
                overlap = len(q & words)
                if overlap:
                    scored.append((overlap, -r.rank, s.strip(), r.rank))
        if not scored:
            return NO_CONTEXT_MESSAGE
        scored.sort(reverse=True)
        picked = scored[: self.max_sentences]
        return " ".join(f"{s} [Izvor {rank}]" for _, _, s, rank in picked)


class OpenAICompatibleGenerator(BaseGenerator):
    """Adapter za udaljeni ili lokalni LLM s OpenAI-kompatibilnim API-jem.

    Konfiguracija preko varijabli okruženja:
      RAG_LLM_BASE_URL  (npr. https://api.openai.com/v1 ili http://localhost:11434/v1)
      RAG_LLM_API_KEY   (za lokalne poslužitelje može biti bilo koja vrijednost)
      RAG_LLM_MODEL     (naziv modela)
    """
    name = "openai-compatible"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, temperature: float = 0.0, timeout: int = 60):
        self.base_url = (base_url or os.environ.get("RAG_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("RAG_LLM_API_KEY", "")
        self.model = model or os.environ.get("RAG_LLM_MODEL", "")
        self.temperature = temperature
        self.timeout = timeout
        if not (self.base_url and self.model):
            raise ValueError("Postavite RAG_LLM_BASE_URL i RAG_LLM_MODEL.")

    def generate(self, question: str, results: List[RetrievalResult]) -> str:
        if not results:
            return NO_CONTEXT_MESSAGE
        payload = {"model": self.model, "temperature": self.temperature,
                   "messages": build_messages(question, format_with_sources(results))}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


def make_generator(kind: str = "extractive", **kwargs) -> BaseGenerator:
    if kind == "extractive":
        return ExtractiveGenerator(**kwargs)
    if kind in ("openai", "llm"):
        return OpenAICompatibleGenerator(**kwargs)
    raise ValueError(f"Nepoznata vrsta generatora: {kind}")
