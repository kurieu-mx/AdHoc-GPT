"""Tokenizers written from scratch: character-level and byte-level BPE.

Both expose the same tiny interface::

    tok.encode("hello") -> list[int]
    tok.decode([1, 2, 3]) -> str
    tok.vocab_size -> int
    tok.save(path) / load_tokenizer(path)

No external tokenizer library is used anywhere in this file.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

# GPT-4 style pre-tokenisation pattern: keeps words, contractions, numbers and
# runs of whitespace as separate chunks so merges never cross word boundaries.
SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)

try:  # `regex` gives us \p{L}; fall back to a plain-re approximation.
    import regex as _re

    _SPLIT = _re.compile(SPLIT_PATTERN)
except ImportError:  # pragma: no cover - exercised only without `regex`
    _SPLIT = re.compile(
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\w]?[^\W\d_]+|\d{1,3}"
        r"| ?[^\s\w]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
    )


class BaseTokenizer:
    """Shared helpers for the tokenizers below."""

    kind: str = "base"

    @property
    def vocab_size(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:  # pragma: no cover - overridden
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    # --- persistence -----------------------------------------------------
    def state_dict(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), ensure_ascii=False, indent=2))


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer -- one id per unique character in the corpus.

    Tiny vocabulary (~65 symbols for Shakespeare), which is exactly what you
    want while debugging the transformer itself.
    """

    kind = "char"

    def __init__(self, chars: Sequence[str]):
        self.itos = list(dict.fromkeys(chars))  # de-dupe, keep order
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        stoi = self.stoi
        # unknown characters are skipped rather than crashing at inference time
        return [stoi[c] for c in text if c in stoi]

    def decode(self, ids: Sequence[int]) -> str:
        itos = self.itos
        return "".join(itos[int(i)] for i in ids if 0 <= int(i) < len(itos))

    def state_dict(self) -> dict:
        return {"kind": self.kind, "itos": self.itos}


class BPETokenizer(BaseTokenizer):
    """Byte-level byte-pair-encoding tokenizer trained from scratch.

    Training works on UTF-8 *bytes*, so every possible string round-trips.
    Text is pre-split with :data:`SPLIT_PATTERN` before merging, which keeps
    merges from spanning word boundaries (the trick GPT-2 uses).
    """

    kind = "bpe"

    def __init__(self, merges: dict[tuple[int, int], int], special_tokens: dict[str, int] | None = None):
        self.merges = merges                      # (a, b) -> new id
        self.special_tokens = dict(special_tokens or {})
        self.vocab = self._build_vocab()
        self._cache: dict[str, list[int]] = {}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(s) for s in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )

    # --- training --------------------------------------------------------
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 1024,
        special_tokens: Sequence[str] = ("<|endoftext|>",),
        verbose: bool = False,
    ) -> "BPETokenizer":
        n_special = len(special_tokens)
        n_merges = vocab_size - 256 - n_special
        if n_merges < 0:
            raise ValueError(f"vocab_size must be >= {256 + n_special}")

        # word -> frequency, each word held as a list of byte ids
        freqs: Counter[tuple[int, ...]] = Counter()
        for chunk in _SPLIT.findall(text):
            freqs[tuple(chunk.encode("utf-8"))] += 1
        words = [list(w) for w in freqs]
        counts_per_word = list(freqs.values())

        # Pair statistics are maintained incrementally: after a merge only the
        # words that contained the merged pair are re-counted, which is what
        # makes training on multi-megabyte corpora practical.
        stats: Counter[tuple[int, int]] = Counter()
        where: dict[tuple[int, int], set[int]] = {}
        for i, (word, c) in enumerate(zip(words, counts_per_word)):
            for pair in zip(word, word[1:]):
                stats[pair] += c
                where.setdefault(pair, set()).add(i)

        merges: dict[tuple[int, int], int] = {}
        for i in range(n_merges):
            pair, count = None, 1
            for p, c in stats.items():
                if c > count:
                    pair, count = p, c
            if pair is None:  # nothing left worth merging
                break
            new_id = 256 + i
            merges[pair] = new_id

            for wi in list(where.get(pair, ())):
                word, c = words[wi], counts_per_word[wi]
                for p in zip(word, word[1:]):
                    stats[p] -= c
                new_word = _merge(word, pair, new_id)
                words[wi] = new_word
                for p in zip(new_word, new_word[1:]):
                    stats[p] += c
                    where.setdefault(p, set()).add(wi)
            del stats[pair]
            where.pop(pair, None)

            if verbose and (i + 1) % 200 == 0:
                print(f"  merge {i + 1}/{n_merges}: {pair} -> {new_id} (count {count})")

        base = 256 + len(merges)
        specials = {tok: base + j for j, tok in enumerate(special_tokens)}
        return cls(merges, specials)

    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            vocab[idx] = vocab[a] + vocab[b]
        for tok, idx in self.special_tokens.items():
            vocab[idx] = tok.encode("utf-8")
        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # --- encoding / decoding --------------------------------------------
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # merge the pair with the lowest merge rank that is still present
            pair = min(
                zip(ids, ids[1:]),
                key=lambda p: self.merges.get(p, float("inf")),
            )
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) < 100_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        if self._special_re is not None and allowed_special:
            out: list[int] = []
            for part in self._special_re.split(text):
                if not part:
                    continue
                if part in self.special_tokens:
                    out.append(self.special_tokens[part])
                else:
                    out.extend(self._encode_ordinary(part))
            return out
        return self._encode_ordinary(text)

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _SPLIT.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        vocab = self.vocab
        parts = [vocab[int(i)] for i in ids if int(i) in vocab]
        return b"".join(parts).decode("utf-8", errors="replace")

    def state_dict(self) -> dict:
        return {
            "kind": self.kind,
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }


def _merge(ids: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out: list[int] = []
    i = 0
    n = len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def load_tokenizer(path: str | Path) -> BaseTokenizer:
    """Load a tokenizer saved by :meth:`BaseTokenizer.save`."""
    state = json.loads(Path(path).read_text())
    kind = state.get("kind", "char")
    if kind == "char":
        return CharTokenizer(state["itos"])
    if kind == "bpe":
        merges = {(a, b): idx for a, b, idx in state["merges"]}
        return BPETokenizer(merges, state.get("special_tokens"))
    raise ValueError(f"unknown tokenizer kind {kind!r}")
