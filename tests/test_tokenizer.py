import torch  # noqa: F401  (ensures the venv is the one under test)

from adhoc_gpt.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer

TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune\n"
) * 20


def test_char_roundtrip():
    tok = CharTokenizer.train(TEXT)
    assert tok.vocab_size == len(set(TEXT))
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_char_skips_unknown():
    tok = CharTokenizer.train("abc")
    assert tok.decode(tok.encode("abcZ")) == "abc"


def test_bpe_roundtrip_and_compression():
    tok = BPETokenizer.train(TEXT, vocab_size=400)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    # merges must actually shorten the sequence versus raw bytes
    assert len(ids) < len(TEXT.encode("utf-8")) * 0.6
    assert tok.vocab_size <= 400


def test_bpe_handles_unicode_and_specials():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    s = "héllo wörld 🌍"
    assert tok.decode(tok.encode(s)) == s
    ids = tok.encode("a<|endoftext|>b")
    assert tok.special_tokens["<|endoftext|>"] in ids


def test_incremental_pair_stats_match_a_naive_recount():
    """The fast trainer must merge the same pair *counts* as a full recount would.

    Which pair wins a tie is arbitrary, but the winning count at every step is not.
    """
    from collections import Counter

    from adhoc_gpt.tokenizer import _SPLIT, _merge

    n_merges = 60
    freqs = Counter(tuple(c.encode("utf-8")) for c in _SPLIT.findall(TEXT))
    words, counts = [list(w) for w in freqs], list(freqs.values())

    naive_counts = []
    for _ in range(n_merges):
        stats = Counter()
        for word, c in zip(words, counts):
            for pair in zip(word, word[1:]):
                stats[pair] += c
        pair, best = None, 1
        for p, c in stats.items():
            if c > best:
                pair, best = p, c
        if pair is None:
            break
        naive_counts.append(best)
        words = [_merge(w, pair, 10**6 + len(naive_counts)) for w in words]

    tok = BPETokenizer.train(TEXT, vocab_size=256 + len(naive_counts) + 1)
    # recompute what each of the fast trainer's merges was worth, in order
    words, counts = [list(w) for w in freqs], list(freqs.values())
    fast_counts = []
    for pair, new_id in sorted(tok.merges.items(), key=lambda kv: kv[1]):
        stats = Counter()
        for word, c in zip(words, counts):
            for p in zip(word, word[1:]):
                stats[p] += c
        fast_counts.append(stats[pair])
        words = [_merge(w, pair, new_id) for w in words]

    assert fast_counts == naive_counts[: len(fast_counts)]


def test_save_load_roundtrip(tmp_path):
    for tok in (CharTokenizer.train(TEXT), BPETokenizer.train(TEXT, vocab_size=350)):
        p = tmp_path / f"{tok.kind}.json"
        tok.save(p)
        loaded = load_tokenizer(p)
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.encode(TEXT) == tok.encode(TEXT)
        assert loaded.decode(loaded.encode(TEXT)) == TEXT
