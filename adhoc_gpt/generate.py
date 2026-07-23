"""Sample text from a trained AdHoc-GPT checkpoint.

Example::

    python -m adhoc_gpt.generate --ckpt runs/mini-adhoc-lm/ckpt.pt \
        --prompt "KING RICHARD III:" --tokens 500 --temperature 0.8 --top-k 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import GPTConfig
from .model import AdHocGPT
from .tokenizer import load_tokenizer
from .train import resolve_device


def load_model(ckpt_path: str | Path, device: str = "auto"):
    """Load a checkpoint plus its tokenizer (looked up from the run's data dir)."""
    device = resolve_device(device)
    ckpt_path = Path(ckpt_path)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig.from_dict(ck["model_config"])
    model = AdHocGPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval().to(device)

    tok_path = ckpt_path.parent / "tokenizer.json"
    if not tok_path.exists():
        data_dir = ck.get("train_config", {}).get("data_dir")
        if data_dir:
            tok_path = Path(data_dir) / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"no tokenizer.json next to {ckpt_path} or in the run's data dir"
        )
    return model, load_tokenizer(tok_path), device


@torch.no_grad()
def sample(
    ckpt: str | Path,
    prompt: str = "",
    tokens: int = 500,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = None,
    num_samples: int = 1,
    seed: int = 1337,
    device: str = "auto",
) -> list[str]:
    model, tok, device = load_model(ckpt, device)
    torch.manual_seed(seed)

    ids = tok.encode(prompt) if prompt else [0]
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

    outputs = []
    for _ in range(num_samples):
        out = model.generate(
            idx, max_new_tokens=tokens, temperature=temperature, top_k=top_k, top_p=top_p
        )
        outputs.append(tok.decode(out[0].tolist()))
    return outputs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate text with AdHoc-GPT")
    p.add_argument("--ckpt", default="runs/mini-adhoc-lm/ckpt.pt")
    p.add_argument("--prompt", default="")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    outs = sample(a.ckpt, a.prompt, a.tokens, a.temperature, a.top_k, a.top_p,
                  a.num_samples, a.seed, a.device)
    for i, text in enumerate(outs):
        if len(outs) > 1:
            print(f"\n===== sample {i + 1} =====")
        print(text)


if __name__ == "__main__":
    main()
