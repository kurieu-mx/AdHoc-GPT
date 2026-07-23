"""Phase 4: retrieval over a clause library, implemented from scratch.

Two retrievers, no external search dependency:

* :class:`BM25Index` -- lexical scoring (Okapi BM25), built over a document
  store of clauses/precedents.
* :class:`EmbeddingIndex` -- dense retrieval that reuses the *model's own*
  token embeddings: a document vector is the mean of its token embeddings,
  scored by cosine similarity. No second model to train.
* :class:`HybridRetriever` -- reciprocal-rank fusion of the two.

The retrieved clauses are formatted into a prompt for the fine-tuned model by
:func:`build_prompt`, which is what the Phase-4 application calls.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "in", "on", "for", "that", "with", "as",
    "by", "is", "are", "be", "its", "it", "at", "from", "or", "this", "which",
    "shall", "all", "any", "such", "their", "we", "our",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Document:
    """One retrievable unit -- typically a single clause."""

    id: str
    text: str
    kind: str = "clause"        # clause | preambular | operative | precedent
    topic: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hit:
    doc: Document
    score: float


class BM25Index:
    """Okapi BM25 (k1/b tunable), built from term statistics over the corpus."""

    def __init__(self, docs: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d.text) for d in self.docs]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.tf: list[Counter] = [Counter(t) for t in self.tokens]
        df: Counter = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = len(self.docs)
        # BM25+ style idf, floored so common terms cannot go negative
        self.idf = {
            term: max(math.log(1 + (n - freq + 0.5) / (freq + 0.5)), 1e-6)
            for term, freq in df.items()
        }
        self.postings: dict[str, list[int]] = {}
        for i, t in enumerate(self.tokens):
            for term in set(t):
                self.postings.setdefault(term, []).append(i)

    def score(self, query_terms: Sequence[str], i: int) -> float:
        tf, dl = self.tf[i], self.lengths[i]
        s = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avg_len, 1e-9))
            s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
        return s

    def search(self, query: str, k: int = 5, kind: str | None = None) -> list[Hit]:
        terms = tokenize(query)
        candidates: set[int] = set()
        for term in terms:
            candidates.update(self.postings.get(term, ()))
        hits = [
            Hit(self.docs[i], self.score(terms, i))
            for i in candidates
            if kind is None or self.docs[i].kind == kind
        ]
        hits.sort(key=lambda h: (-h.score, h.doc.id))
        return hits[:k]


class EmbeddingIndex:
    """Dense retrieval using the language model's own token embedding matrix."""

    def __init__(self, docs: Sequence[Document], model, tokenizer):
        import torch

        self.docs = list(docs)
        self.model, self.tokenizer = model, tokenizer
        with torch.no_grad():
            self.matrix = torch.stack([self._embed(d.text) for d in self.docs])

    def _embed(self, text: str):
        import torch

        ids = self.tokenizer.encode(text)[: self.model.config.block_size]
        if not ids:
            return torch.zeros(self.model.config.n_embd)
        wte = self.model.transformer.wte.weight.detach().cpu()
        vec = wte[torch.tensor(ids)].mean(dim=0)
        return vec / vec.norm().clamp_min(1e-8)

    def search(self, query: str, k: int = 5, kind: str | None = None) -> list[Hit]:
        import torch

        with torch.no_grad():
            q = self._embed(query)
            sims = (self.matrix @ q).tolist()
        hits = [
            Hit(d, s) for d, s in zip(self.docs, sims) if kind is None or d.kind == kind
        ]
        hits.sort(key=lambda h: (-h.score, h.doc.id))
        return hits[:k]


class HybridRetriever:
    """Reciprocal-rank fusion: score = sum(1 / (rrf_k + rank)) over retrievers."""

    def __init__(self, *retrievers, rrf_k: int = 60):
        self.retrievers = [r for r in retrievers if r is not None]
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5, kind: str | None = None) -> list[Hit]:
        fused: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, k=k * 3, kind=kind)):
                fused[hit.doc.id] = fused.get(hit.doc.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                docs[hit.doc.id] = hit.doc
        hits = [Hit(docs[i], s) for i, s in fused.items()]
        hits.sort(key=lambda h: (-h.score, h.doc.id))
        return hits[:k]


def mmr(hits: Sequence[Hit], k: int = 4, lam: float = 0.7) -> list[Hit]:
    """Maximal marginal relevance: trade relevance against redundancy.

    Templated precedent produces many near-identical clauses; picking the top-k
    by score alone returns the same sentence with four different opening
    participles. Similarity is Jaccard overlap of content terms.
    """
    if not hits:
        return []
    top = max(h.score for h in hits) or 1.0
    sets = {h.doc.id: set(tokenize(h.doc.text)) for h in hits}
    selected: list[Hit] = []
    pool = list(hits)
    while pool and len(selected) < k:
        best, best_val = None, -1e9
        for h in pool:
            rel = h.score / top
            red = max(
                (len(sets[h.doc.id] & sets[s.doc.id])
                 / max(len(sets[h.doc.id] | sets[s.doc.id]), 1) for s in selected),
                default=0.0,
            )
            val = lam * rel - (1 - lam) * red
            if val > best_val:
                best, best_val = h, val
        selected.append(best)
        pool.remove(best)
    return selected


# --------------------------------------------------------------------------
# clause library
# --------------------------------------------------------------------------
CLAUSE_SPLIT = re.compile(r"\n\s*\n")


def clauses_from_corpus(
    text: str, max_docs: int | None = None, dedupe: bool = True
) -> list[Document]:
    """Split a diplomacy corpus into individually retrievable clauses.

    Templated corpora repeat clauses verbatim; ``dedupe`` keeps one copy of each
    so retrieval returns four *different* precedents instead of four identical ones.
    """
    docs: list[Document] = []
    seen: set[str] = set()
    for d_i, doc in enumerate(text.split("<|endoftext|>")):
        doc = doc.strip()
        if not doc:
            continue
        topic = ""
        m = re.search(r"(?:Title: Resolution on|Agenda:|Topic:|Topic under consideration:)\s*(.+)",
                      doc)
        if m:
            topic = m.group(1).strip().rstrip(".")
        source = doc.splitlines()[0][:80] if doc.splitlines() else ""
        for c_i, chunk in enumerate(CLAUSE_SPLIT.split(doc)):
            chunk = " ".join(chunk.split())
            if len(chunk) < 40:
                continue
            key = re.sub(r"^\d+\.\s*", "", chunk).lower()
            if dedupe:
                if key in seen:
                    continue
                seen.add(key)
            if re.match(r"^\d+\.\s", chunk):
                kind = "operative"
            elif chunk.endswith(",") and chunk[0].isupper():
                kind = "preambular"
            elif chunk.startswith("DELEGATE") or "Chair" in chunk[:30]:
                kind = "debate"
            else:
                kind = "clause"
            docs.append(Document(f"d{d_i}c{c_i}", chunk, kind, topic, source))
        if max_docs and len(docs) >= max_docs:
            break
    return docs


def save_library(docs: Iterable[Document], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([d.to_dict() for d in docs], indent=1))
    return path


def load_library(path: str | Path) -> list[Document]:
    return [Document(**d) for d in json.loads(Path(path).read_text())]


# --------------------------------------------------------------------------
# prompt assembly
# --------------------------------------------------------------------------
def build_prompt(
    task: str,
    hits: Sequence[Hit],
    organ: str = "The General Assembly",
    max_clause_chars: int = 220,
) -> str:
    """Format retrieved precedent + the drafting task into a model prompt.

    The layout mirrors the corpus, so a model fine-tuned on the corpus sees a
    familiar document header and continues it rather than describing it. Long
    clauses are trimmed -- a small model's context is the scarce resource here.
    """
    lines = [f"Title: Resolution on {task}", ""]
    if hits:
        lines.append("Precedent clauses retrieved for reference:")
        for h in hits:
            text = h.doc.text
            if len(text) > max_clause_chars:
                text = text[:max_clause_chars].rsplit(" ", 1)[0] + " ..."
            lines.append(f"- {text}")
        lines.append("")
    lines += [f"{organ},", ""]
    return "\n".join(lines)


def fit_prompt(
    task: str,
    hits: Sequence[Hit],
    tokenizer,
    budget: int,
    organ: str = "The General Assembly",
) -> tuple[str, list[Hit]]:
    """Build the longest prompt that fits ``budget`` tokens, dropping clauses.

    Returns the prompt and the clauses that actually made it in, so the caller
    reports what the model really saw rather than what was retrieved.
    """
    task = " ".join(task.split())
    # A pathological task string can overflow the context on its own, before any
    # precedent is added. Shrink it -- but never below a length that a genuine
    # topic could need, so a small context truncates the *precedent*, not the ask.
    while len(task) > 60 and len(tokenizer.encode(build_prompt(task, [], organ))) > budget:
        task = task[: max(60, len(task) // 2)].rstrip()

    kept = list(hits)
    while kept:
        prompt = build_prompt(task, kept, organ)
        if len(tokenizer.encode(prompt)) <= budget:
            return prompt, kept
        kept.pop()
    return build_prompt(task, [], organ), []


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build / query the clause library")
    p.add_argument("--corpus", default="data/raw/diplomacy.txt")
    p.add_argument("--library", default="data/clause_library.json")
    p.add_argument("--build", action="store_true", help="(re)build the library from the corpus")
    p.add_argument("--query", default=None)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--kind", default=None)
    a = p.parse_args()

    if a.build:
        docs = clauses_from_corpus(Path(a.corpus).read_text(encoding="utf-8"))
        save_library(docs, a.library)
        kinds = Counter(d.kind for d in docs)
        print(f"{len(docs):,} clauses -> {a.library}  ({dict(kinds)})")
    if a.query:
        docs = load_library(a.library)
        for hit in BM25Index(docs).search(a.query, a.k, a.kind):
            print(f"[{hit.score:5.2f}] ({hit.doc.kind}) {hit.doc.text[:160]}")


if __name__ == "__main__":
    _cli()
