"""Model and training configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path


@dataclass
class GPTConfig:
    """Architecture hyper-parameters for a decoder-only transformer."""

    vocab_size: int = 65          # set by the tokenizer at prepare time
    block_size: int = 256         # maximum context length in tokens
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 320
    dropout: float = 0.2
    bias: bool = False            # bias in Linear / LayerNorm (GPT-2 uses True)
    tie_weights: bool = True      # share token embedding with the output head

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "GPTConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrainConfig:
    """Optimisation hyper-parameters."""

    # data
    data_dir: str = "data/shakespeare_char"
    out_dir: str = "runs/mini-adhoc-lm"

    # optimisation
    batch_size: int = 64
    grad_accum_steps: int = 1
    max_iters: int = 5000
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 200
    lr_decay_iters: int = 0       # 0 -> defaults to max_iters
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0

    # evaluation / logging
    eval_interval: int = 250
    eval_iters: int = 100
    log_interval: int = 10
    always_save_checkpoint: bool = False

    # runtime
    device: str = "auto"          # auto | cuda | cpu | mps
    dtype: str = "auto"           # auto | bfloat16 | float16 | float32
    compile: bool = False
    seed: int = 1337
    flash: bool = True            # use fused SDPA kernel instead of the manual path

    def resolved_lr_decay_iters(self) -> int:
        return self.lr_decay_iters or self.max_iters


#: Ready-made model sizes. ``mini`` is the Phase-1 target (5-10M parameters).
PRESETS: dict[str, dict] = {
    "nano": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.1),
    "mini": dict(n_layer=6, n_head=8, n_embd=320, block_size=256, dropout=0.2),
    "small": dict(n_layer=8, n_head=8, n_embd=512, block_size=512, dropout=0.1),
    "base": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, dropout=0.0),
}


def preset_config(name: str, **overrides) -> GPTConfig:
    """Build a :class:`GPTConfig` from a named preset with optional overrides."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = dict(PRESETS[name])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return GPTConfig(**cfg)
