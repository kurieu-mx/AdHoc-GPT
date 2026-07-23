"""AdHoc-GPT: a decoder-only transformer built from scratch in PyTorch."""

from .config import GPTConfig, TrainConfig, PRESETS
from .model import AdHocGPT
from .tokenizer import CharTokenizer, BPETokenizer, load_tokenizer

__version__ = "0.1.0"

__all__ = [
    "GPTConfig",
    "TrainConfig",
    "PRESETS",
    "AdHocGPT",
    "CharTokenizer",
    "BPETokenizer",
    "load_tokenizer",
]
