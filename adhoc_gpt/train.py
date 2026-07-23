"""Training loop for AdHoc-GPT.

Single GPU / CPU::

    python -m adhoc_gpt.train --data-dir data/shakespeare_char --preset mini

Multi-GPU (Phase 2, data-parallel via torchrun)::

    torchrun --standalone --nproc_per_node=4 -m adhoc_gpt.train --preset small

Writes ``ckpt.pt`` (best val loss), ``last.pt``, the two config JSONs and a
``metrics.csv`` of the loss curves into ``--out-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import GPTConfig, TrainConfig, preset_config
from .data import BinDataset, load_dataset_tokenizer
from .model import AdHocGPT


# --------------------------------------------------------------------------
# runtime helpers
# --------------------------------------------------------------------------
def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str) -> str:
    if name != "auto":
        return name
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16" if device.startswith("cuda") else "float32"


class DistInfo:
    """Data-parallel context, populated from the environment torchrun sets up."""

    def __init__(self):
        self.enabled = int(os.environ.get("RANK", -1)) != -1
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_master = self.rank == 0

    def setup(self, device: str) -> str:
        if not self.enabled:
            return device
        backend = "nccl" if device.startswith("cuda") else "gloo"
        init_process_group(backend=backend)
        if device.startswith("cuda"):
            device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(device)
        return device

    def teardown(self) -> None:
        if self.enabled:
            destroy_process_group()


def get_lr(it: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to ``min_lr``."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / max(cfg.warmup_iters, 1)
    decay_iters = cfg.resolved_lr_decay_iters()
    if it > decay_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(decay_iters - cfg.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, splits: dict[str, BinDataset], cfg: TrainConfig, device: str, ctx) -> dict:
    """Average the loss over ``eval_iters`` batches of each split."""
    out = {}
    model.eval()
    for name, ds in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


# --------------------------------------------------------------------------
# main training loop
# --------------------------------------------------------------------------
def train(
    model_cfg: GPTConfig,
    cfg: TrainConfig,
    resume: bool = False,
    sample_every: int = 0,
    init_from: str | Path | None = None,
) -> dict:
    """Train (or fine-tune, via ``init_from``) a model and return a run summary."""
    dist = DistInfo()
    device = dist.setup(resolve_device(cfg.device))
    master = dist.is_master

    torch.manual_seed(cfg.seed + dist.rank)  # different data order per rank
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if dist.enabled:
        if cfg.grad_accum_steps % dist.world_size != 0:
            raise ValueError(
                f"grad_accum_steps ({cfg.grad_accum_steps}) must be divisible by "
                f"world_size ({dist.world_size})"
            )
        cfg.grad_accum_steps //= dist.world_size

    device_type = "cuda" if device.startswith("cuda") else device
    dtype = resolve_dtype(cfg.dtype, device)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
        nullcontext()
        if device_type != "cuda"
        else torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    )

    out_dir = Path(cfg.out_dir)
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- fine-tuning: the checkpoint's architecture wins, including block_size,
    # so this must happen before the datasets are built ---
    base = None
    if init_from is not None:
        base = torch.load(init_from, map_location="cpu", weights_only=False)
        base_cfg = GPTConfig.from_dict(base["model_config"])
        base_cfg.dropout = model_cfg.dropout  # dropout may be re-tuned for fine-tuning
        model_cfg = base_cfg

    # --- data ---
    train_ds = BinDataset(cfg.data_dir, "train", model_cfg.block_size)
    val_ds = BinDataset(cfg.data_dir, "val", model_cfg.block_size)
    splits = {"train": train_ds, "val": val_ds}

    # --- model ---
    if base is not None:
        if model_cfg.vocab_size != train_ds.vocab_size:
            raise ValueError(
                f"checkpoint vocab_size {model_cfg.vocab_size} != dataset vocab_size "
                f"{train_ds.vocab_size}. Fine-tuning needs the tokenizer the base model "
                f"was trained with -- re-run data prep with --tokenizer-from <that "
                f"tokenizer.json>."
            )
        model = AdHocGPT(model_cfg)
        model.load_state_dict(base["model"])
        if master:
            print(f"initialised from {init_from} (iter {base.get('iter')}, "
                  f"val {base.get('best_val_loss', float('nan')):.4f})")
    else:
        model_cfg.vocab_size = train_ds.vocab_size  # authoritative: from the tokenizer
        model = AdHocGPT(model_cfg)
    model.set_flash(cfg.flash)
    model.to(device)
    n_params = model.num_params()
    tokens_per_iter = cfg.batch_size * cfg.grad_accum_steps * model_cfg.block_size * dist.world_size
    if master:
        print(
            f"device={device} dtype={dtype} | params={n_params/1e6:.2f}M "
            f"(non-emb {model.num_params(True)/1e6:.2f}M) | vocab={model_cfg.vocab_size} "
            f"| block={model_cfg.block_size}"
            + (f" | DDP world_size={dist.world_size}" if dist.enabled else "")
        )
        print(f"tokens/iter={tokens_per_iter:,} | train tokens={train_ds.n_tokens:,} "
              f"| epochs={cfg.max_iters * tokens_per_iter / train_ds.n_tokens:.1f}")

    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type
    )
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))

    start_iter, best_val = 0, float("inf")
    if resume and (out_dir / "last.pt").exists():
        ck = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_iter, best_val = ck["iter"] + 1, ck.get("best_val_loss", float("inf"))
        if master:
            print(f"resumed from iter {start_iter} (best val {best_val:.4f})")

    raw_model = model
    if cfg.compile and hasattr(torch, "compile"):
        if master:
            print("compiling the model (torch.compile) ...")
        model = torch.compile(model)
    if dist.enabled:
        model = DDP(model, device_ids=[dist.local_rank] if device_type == "cuda" else None)

    writer = metrics_file = None
    if master:
        model_cfg.to_json(out_dir / "model_config.json")
        (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))
        metrics_path = out_dir / "metrics.csv"
        new_file = not (resume and metrics_path.exists())
        metrics_file = metrics_path.open("w" if new_file else "a", newline="")
        writer = csv.writer(metrics_file)
        if new_file:
            writer.writerow(["iter", "tokens", "train_loss", "val_loss", "lr", "elapsed_s", "mfu"])

    tokenizer = None
    if sample_every and master:
        try:
            tokenizer = load_dataset_tokenizer(cfg.data_dir)
        except Exception as e:  # pragma: no cover
            print(f"(no tokenizer for sampling: {e})")

    if master:
        print(f"training for {cfg.max_iters} iters -> {out_dir}")
    x, y = train_ds.get_batch(cfg.batch_size, device)
    t_start = time.time()
    t0 = time.time()
    iters_timed = 0
    running_mfu = -1.0
    history: list[dict] = []

    for it in range(start_iter, cfg.max_iters):
        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # --- evaluate / checkpoint (master only) ---
        if (it % cfg.eval_interval == 0 or it == cfg.max_iters - 1) and master:
            losses = estimate_loss(raw_model, splits, cfg, device, ctx)
            elapsed = time.time() - t_start
            print(
                f"iter {it:>6} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr:.2e} | {elapsed:.0f}s"
            )
            writer.writerow([it, it * tokens_per_iter, f"{losses['train']:.6f}",
                             f"{losses['val']:.6f}", f"{lr:.6e}", f"{elapsed:.2f}",
                             f"{max(running_mfu, 0):.4f}"])
            metrics_file.flush()
            history.append({"iter": it, **losses})

            if losses["val"] < best_val or cfg.always_save_checkpoint:
                best_val = min(best_val, losses["val"])
                ckpt = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": asdict(model_cfg),
                    "iter": it,
                    "best_val_loss": best_val,
                    "train_config": asdict(cfg),
                }
                torch.save(ckpt, out_dir / "ckpt.pt")

            if tokenizer is not None and it > start_iter:
                ctxt = torch.zeros((1, 1), dtype=torch.long, device=device)
                sample = raw_model.generate(ctxt, 200, temperature=0.8, top_k=50)
                print("--- sample ---")
                print(tokenizer.decode(sample[0].tolist()).strip()[:400])
                print("--------------")

            t0, iters_timed = time.time(), 0  # don't charge evaluation time to training

        # --- forward / backward with gradient accumulation ---
        for micro in range(cfg.grad_accum_steps):
            if dist.enabled:
                # only all-reduce gradients on the last micro-step
                model.require_backward_grad_sync = micro == cfg.grad_accum_steps - 1
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            x, y = train_ds.get_batch(cfg.batch_size, device)  # prefetch next batch
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        iters_timed += 1
        if it % cfg.log_interval == 0 and it > start_iter and master:
            # CUDA kernels are queued asynchronously: without this sync the timer
            # measures launch overhead, not compute (and MFU comes out >100%).
            if device_type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) / max(iters_timed, 1)  # mean seconds per iteration
            t0, iters_timed = t1, 0
            lossf = loss.item() * cfg.grad_accum_steps
            mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.grad_accum_steps, dt)
            running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f"iter {it:>6} | loss {lossf:.4f} | {dt*1000:.1f} ms/iter "
                f"| mfu {running_mfu*100:.1f}%"
            )

    # --- final save (master only) ---
    total = time.time() - t_start
    summary = {
        "params": n_params,
        "best_val_loss": best_val,
        "final": history[-1] if history else {},
        "train_time_s": round(total, 1),
        "tokens_seen": cfg.max_iters * tokens_per_iter,
        "out_dir": str(out_dir),
    }
    if master:
        torch.save(
            {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(model_cfg),
                "iter": cfg.max_iters - 1,
                "best_val_loss": best_val,
                "train_config": asdict(cfg),
            },
            out_dir / "last.pt",
        )
        metrics_file.close()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"done in {total/60:.1f} min | best val loss {best_val:.4f} | -> {out_dir}")
    dist.teardown()
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train AdHoc-GPT")
    p.add_argument("--preset", default="mini", choices=["nano", "mini", "small", "base"])
    p.add_argument("--data-dir", default="data/shakespeare_char")
    p.add_argument("--out-dir", default="runs/mini-adhoc-lm")
    # architecture overrides
    p.add_argument("--n-layer", type=int)
    p.add_argument("--n-head", type=int)
    p.add_argument("--n-embd", type=int)
    p.add_argument("--block-size", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--bias", action="store_true", help="use biases in Linear/LayerNorm")
    # optimisation
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # eval / runtime
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=100)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-flash", action="store_true", help="use the manual attention maths")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", default=None,
                   help="checkpoint to initialise from (fine-tuning); architecture is "
                        "taken from the checkpoint")
    p.add_argument("--sample-every-eval", action="store_true",
                   help="print a sample at every evaluation")
    return p


def main(argv: list[str] | None = None) -> dict:
    a = build_parser().parse_args(argv)
    model_cfg = preset_config(
        a.preset,
        n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd,
        block_size=a.block_size, dropout=a.dropout,
        bias=True if a.bias else None,
    )
    cfg = TrainConfig(
        data_dir=a.data_dir, out_dir=a.out_dir,
        batch_size=a.batch_size, grad_accum_steps=a.grad_accum_steps,
        max_iters=a.max_iters, learning_rate=a.lr, min_lr=a.min_lr,
        warmup_iters=a.warmup_iters, weight_decay=a.weight_decay, grad_clip=a.grad_clip,
        eval_interval=a.eval_interval, eval_iters=a.eval_iters, log_interval=a.log_interval,
        device=a.device, dtype=a.dtype, seed=a.seed, compile=a.compile, flash=not a.no_flash,
    )
    return train(model_cfg, cfg, resume=a.resume, sample_every=int(a.sample_every_eval),
                 init_from=a.init_from)


if __name__ == "__main__":
    main()
