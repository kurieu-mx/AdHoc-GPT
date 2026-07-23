import random
import re

import pytest

from adhoc_gpt.domain.corpus import (
    OPERATIVE_FRAMES,
    OPERATIVE_VERBS,
    PREAMBULAR_FRAMES,
    PREAMBULAR_OPENERS,
    TOPICS,
    build_corpus,
    make_debate,
    make_position_paper,
    make_resolution,
)
from adhoc_gpt.rag import (
    BM25Index,
    Document,
    Hit,
    HybridRetriever,
    build_prompt,
    fit_prompt,
    clauses_from_corpus,
    load_library,
    mmr,
    save_library,
    tokenize,
)


# --- corpus ---------------------------------------------------------------
def test_clause_banks_are_type_aligned():
    """Every verb type must have frames it can govern, and vice versa."""
    assert set(OPERATIVE_VERBS) == set(OPERATIVE_FRAMES)
    assert set(PREAMBULAR_OPENERS) == set(PREAMBULAR_FRAMES)
    for kind, frames in {**OPERATIVE_FRAMES, **PREAMBULAR_FRAMES}.items():
        assert frames, f"no frames for {kind}"


def test_resolution_has_the_expected_skeleton():
    text = make_resolution(random.Random(0), "climate resilience")
    assert "Title: Resolution on climate resilience" in text
    assert re.search(r"^1\. ", text, re.M), "operative paragraphs are numbered"
    assert text.rstrip().endswith("abstentions.") or text.rstrip().endswith("abstention.")
    # preambular clauses end in a comma, operative ones in ; or .
    body = text.split("\n\n")
    assert any(p.endswith(",") for p in body)


def test_generation_is_deterministic_per_seed():
    a = build_corpus(5, seed=42)
    b = build_corpus(5, seed=42)
    c = build_corpus(5, seed=43)
    assert a == b and a != c


def test_all_topics_render_all_document_kinds():
    rng = random.Random(1)
    for topic in TOPICS:
        for maker in (make_resolution, make_debate, make_position_paper):
            text = maker(rng, topic)
            assert "{" not in text and "}" not in text, f"unfilled template in {maker.__name__}"
            assert len(text) > 200


def test_corpus_documents_are_separated():
    corpus = build_corpus(12, seed=3)
    assert corpus.count("<|endoftext|>") == 11
    assert len(corpus) > 12_000


# --- retrieval ------------------------------------------------------------
@pytest.fixture(scope="module")
def library():
    return clauses_from_corpus(build_corpus(120, seed=5))


def test_tokenize_drops_stopwords_and_case():
    assert "the" not in tokenize("The Security Council")
    assert tokenize("Security COUNCIL") == ["security", "council"]


def test_clauses_are_classified_and_deduped(library):
    kinds = {d.kind for d in library}
    assert {"preambular", "operative"} <= kinds
    texts = [re.sub(r"^\d+\.\s*", "", d.text).lower() for d in library]
    assert len(texts) == len(set(texts)), "dedupe should leave no repeats"
    assert all(len(d.text) >= 40 for d in library)


def test_bm25_ranks_topical_clauses_first(library):
    hits = BM25Index(library).search("nuclear disarmament test ban treaty", k=5)
    assert hits, "expected matches"
    joined = " ".join(h.doc.text.lower() for h in hits)
    assert "nuclear" in joined or "test-ban" in joined
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_bm25_filters_by_kind(library):
    hits = BM25Index(library).search("States shall strengthen", k=5, kind="operative")
    assert hits and all(h.doc.kind == "operative" for h in hits)


def test_bm25_handles_unmatched_query(library):
    assert BM25Index(library).search("xyzzy quuxfoo", k=5) == []


def test_mmr_reduces_redundancy():
    docs = [
        Document("1", "Recalling the contribution of early warning systems to coastal defence"),
        Document("2", "Reaffirming the contribution of early warning systems to coastal defence"),
        Document("3", "Welcoming the contribution of early warning systems to coastal defence"),
        Document("4", "Urging Member States to finance smallholder credit facilities"),
    ]
    hits = BM25Index(docs).search("contribution of early warning systems", k=4)
    picked = mmr(hits, k=2)
    assert len(picked) == 2
    assert picked[0].doc.id != picked[1].doc.id
    # the second pick should be the *different* clause, not another paraphrase
    assert picked[1].doc.id == "4" or len(hits) < 4


def test_hybrid_fusion_merges_rankings(library):
    bm25 = BM25Index(library)

    class Reversed:  # stand-in for a second retriever with a different ranking
        def search(self, query, k=5, kind=None):
            return list(reversed(bm25.search(query, k=k, kind=kind)))

    fused = HybridRetriever(bm25, Reversed()).search("refugee protection", k=4)
    assert len(fused) <= 4
    assert len({h.doc.id for h in fused}) == len(fused), "no duplicates after fusion"


def test_library_roundtrip(tmp_path, library):
    p = save_library(library[:50], tmp_path / "lib.json")
    loaded = load_library(p)
    assert [d.to_dict() for d in loaded] == [d.to_dict() for d in library[:50]]


def test_build_prompt_includes_precedent_and_header(library):
    hits = BM25Index(library).search("cybersecurity critical infrastructure", k=3)
    prompt = build_prompt("cybersecurity", hits, organ="The Security Council")
    assert "Title: Resolution on cybersecurity" in prompt
    assert prompt.rstrip().endswith("The Security Council,")
    for h in hits:
        assert h.doc.text[:100] in prompt


def test_build_prompt_trims_long_clauses(library):
    long_doc = Document("x", "Recalling " + "the situation " * 60)
    prompt = build_prompt("x", [Hit(long_doc, 1.0)], max_clause_chars=100)
    clause_line = [ln for ln in prompt.splitlines() if ln.startswith("- ")][0]
    assert len(clause_line) < 130 and clause_line.endswith("...")


def test_fit_prompt_respects_the_token_budget(library):
    class FakeTok:
        def encode(self, text):
            return list(range(len(text) // 4))  # ~4 chars per token

    hits = BM25Index(library).search("climate resilience adaptation finance", k=6)
    assert len(hits) > 1
    prompt, kept = fit_prompt("climate resilience", hits, FakeTok(), budget=40)
    assert len(FakeTok().encode(prompt)) <= 40
    assert len(kept) < len(hits), "clauses should be dropped to fit"
    assert prompt.rstrip().endswith("The General Assembly,")


def test_fit_prompt_degrades_to_header_only():
    class TinyBudgetTok:
        def encode(self, text):
            return list(range(len(text)))

    hits = [Hit(Document("a", "Recalling something at length " * 5), 1.0)]
    prompt, kept = fit_prompt("topic", hits, TinyBudgetTok(), budget=10)
    assert kept == [] and "Precedent" not in prompt
