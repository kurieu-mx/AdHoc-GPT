"""Dataset preparation and batching.

``prepare`` downloads (or reads) a corpus, trains a tokenizer on it, and writes
``train.bin`` / ``val.bin`` as flat uint16/uint32 token arrays plus a
``meta.json``. Training then memory-maps those files, so the corpus never has
to fit in RAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen

import numpy as np
import torch

from .tokenizer import BPETokenizer, CharTokenizer, load_tokenizer

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

#: corpora that can be fetched without any extra dependency
DIRECT_DATASETS = {"shakespeare": SHAKESPEARE_URL}

#: corpora pulled through HuggingFace ``datasets`` (optional dependency)
HF_DATASETS = {
    "tinystories": ("roneneldan/TinyStories", None, "text"),
    "wikitext": ("wikitext", "wikitext-103-raw-v1", "text"),
}


def download_text(name: str, cache_dir: str | Path = "data/raw", max_docs: int | None = None) -> str:
    """Return the raw text of a supported corpus, caching it on disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    if name in DIRECT_DATASETS:
        print(f"downloading {name} ...")
        with urlopen(DIRECT_DATASETS[name]) as r:
            text = r.read().decode("utf-8")
    elif name in HF_DATASETS:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                f"'{name}' needs the datasets package: pip install datasets"
            ) from e
        path, cfg, col = HF_DATASETS[name]
        split = "train" if max_docs is None else f"train[:{max_docs}]"
        ds = load_dataset(path, cfg, split=split)
        text = "\n".join(ds[col])
    else:
        raise KeyError(
            f"unknown dataset {name!r}; known: {sorted(DIRECT_DATASETS) + sorted(HF_DATASETS)}"
            " (or pass a path to a .txt file)"
        )

    cache.write_text(text, encoding="utf-8")
    return text


def train_shared_tokenizer(
    sources: Sequence[str | Path],
    out: str | Path,
    kind: str = "bpe",
    vocab_size: int = 2048,
    per_source_chars: int = 3_000_000,
) -> dict:
    """Train one tokenizer over several corpora and save it.

    A pretraining corpus and a fine-tuning corpus must share a vocabulary, or
    the fine-tuned model's embedding table means something different from the
    one it was initialised with.
    """
    chunks = []
    for src in sources:
        p = Path(src)
        text = p.read_text(encoding="utf-8") if p.suffix == ".txt" and p.exists() \
            else download_text(str(src))
        chunks.append(text[:per_source_chars])
        print(f"  {src}: {len(text):,} chars (sampling {len(chunks[-1]):,})")
    union = "\n".join(chunks)

    if kind == "char":
        tok = CharTokenizer.train(union)
    else:
        print(f"training byte-level BPE to vocab_size={vocab_size} on {len(union):,} chars ...")
        tok = BPETokenizer.train(union, vocab_size=vocab_size, verbose=True)
    tok.save(out)
    print(f"vocab {tok.vocab_size:,} -> {out}")
    return {"vocab_size": tok.vocab_size, "path": str(out), "kind": tok.kind}


def prepare(
    dataset: str = "shakespeare",
    out_dir: str | Path | None = None,
    tokenizer: str = "char",
    vocab_size: int = 1024,
    val_split: float = 0.1,
    max_docs: int | None = None,
    tokenizer_from: str | Path | None = None,
    tokenizer_sample_chars: int = 5_000_000,
) -> dict:
    """Tokenize a corpus into ``train.bin`` / ``val.bin`` + ``meta.json``.

    ``tokenizer_from`` reuses an existing ``tokenizer.json`` instead of training
    a new one -- required when a pretraining corpus and a fine-tuning corpus
    must share a vocabulary.
    """
    src = Path(dataset)
    if src.suffix == ".txt" and src.exists():
        text, name = src.read_text(encoding="utf-8"), src.stem
    else:
        text, name = download_text(dataset, max_docs=max_docs), dataset

    out_dir = Path(out_dir or f"data/{name}_{tokenizer}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"corpus: {len(text):,} characters")
    if tokenizer_from is not None:
        tok = load_tokenizer(tokenizer_from)
        print(f"reusing tokenizer {tokenizer_from} (vocab {tok.vocab_size:,})")
    elif tokenizer == "char":
        tok = CharTokenizer.train(text)
    elif tokenizer == "bpe":
        # BPE training cost grows with corpus size; a sample is enough to learn merges
        sample = text if len(text) <= tokenizer_sample_chars else text[:tokenizer_sample_chars]
        print(f"training byte-level BPE to vocab_size={vocab_size} "
              f"on {len(sample):,} characters ...")
        tok = BPETokenizer.train(sample, vocab_size=vocab_size, verbose=True)
    else:
        raise ValueError(f"tokenizer must be 'char' or 'bpe', got {tokenizer!r}")
    tok.save(out_dir / "tokenizer.json")
    print(f"vocab size: {tok.vocab_size:,}")

    n = len(text)
    split_at = int(n * (1 - val_split))
    dtype = np.uint16 if tok.vocab_size < 2**16 else np.uint32
    counts = {}
    for label, chunk in (("train", text[:split_at]), ("val", text[split_at:])):
        ids = np.array(tok.encode(chunk), dtype=dtype)
        ids.tofile(out_dir / f"{label}.bin")
        counts[label] = int(ids.size)
        print(f"{label}: {ids.size:,} tokens -> {out_dir / f'{label}.bin'}")

    meta = {
        "dataset": name,
        "tokenizer": tok.kind,
        "vocab_size": tok.vocab_size,
        "dtype": np.dtype(dtype).name,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "compression": round(len(text) / max(counts["train"] + counts["val"], 1), 3),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class BinDataset:
    """Memory-mapped token stream that yields random (x, y) training batches."""

    def __init__(self, data_dir: str | Path, split: str = "train", block_size: int = 256):
        self.dir = Path(data_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.dtype = np.dtype(self.meta.get("dtype", "uint16"))
        self.block_size = block_size
        self.path = self.dir / f"{split}.bin"
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} missing -- run `python -m adhoc_gpt.data` first")
        self.n_tokens = self.path.stat().st_size // self.dtype.itemsize
        if self.n_tokens <= block_size + 1:
            raise ValueError(f"{self.path} has too few tokens for block_size={block_size}")

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def _data(self) -> np.ndarray:
        # re-opened per batch: the documented way to avoid a memmap leak
        return np.memmap(self.path, dtype=self.dtype, mode="r")

    def get_batch(
        self, batch_size: int, device: str = "cpu", generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._data()
        ix = torch.randint(
            len(data) - self.block_size - 1, (batch_size,), generator=generator
        )
        x = torch.stack(
            [torch.from_numpy(data[i : i + self.block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix]
        )
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def load_dataset_tokenizer(data_dir: str | Path):
    return load_tokenizer(Path(data_dir) / "tokenizer.json")


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Tokenize a corpus into train/val .bin files")
    p.add_argument("--dataset", default="shakespeare",
                   help="shakespeare | tinystories | wikitext | path/to/file.txt")
    p.add_argument("--tokenizer", default="char", choices=["char", "bpe"])
    p.add_argument("--vocab-size", type=int, default=1024, help="BPE vocabulary size")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-docs", type=int, default=None,
                   help="cap on HuggingFace rows (keeps TinyStories/WikiText prep quick)")
    p.add_argument("--tokenizer-from", default=None,
                   help="reuse an existing tokenizer.json (shared vocab for fine-tuning)")
    p.add_argument("--train-tokenizer", nargs="+", default=None,
                   help="train a shared tokenizer over these corpora and exit "
                        "(writes to --out-dir/tokenizer.json)")
    a = p.parse_args()

    if a.train_tokenizer:
        out = Path(a.out_dir or "data/shared_vocab") / "tokenizer.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(train_shared_tokenizer(
            a.train_tokenizer, out, a.tokenizer, a.vocab_size), indent=2))
        return

    meta = prepare(a.dataset, a.out_dir, a.tokenizer, a.vocab_size, a.val_split, a.max_docs,
                   a.tokenizer_from)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    _cli()
