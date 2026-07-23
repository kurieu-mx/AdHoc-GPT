"""Phase 2 engineering: throughput benchmarking.

Measures training step time, tokens/s, model-flops utilisation and peak memory
for a set of configurations, so scaling decisions are made on numbers rather
than vibes::

    python -m adhoc_gpt.bench --preset mini --batch-size 64
    python -m adhoc_gpt.bench --preset mini --sweep attention   # flash vs manual
    python -m adhoc_gpt.bench --sweep presets --steps 20
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .config import PRESETS, preset_config
from .model import AdHocGPT
from .train import resolve_device, resolve_dtype


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark(
    cfg,
    batch_size: int = 32,
    steps: int = 20,
    warmup: int = 5,
    device: str = "auto",
    dtype: str = "auto",
    flash: bool = True,
    compile_model: bool = False,
    label: str = "",
) -> dict:
    """Time ``steps`` full training iterations (forward + backward + step)."""
    device = resolve_device(device)
    device_type = "cuda" if device.startswith("cuda") else device
    dtype = resolve_dtype(dtype, device)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
        nullcontext() if device_type != "cuda"
        else torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    )

    raw = AdHocGPT(cfg)
    raw.set_flash(flash)
    raw.to(device)
    opt = raw.configure_optimizers(0.1, 1e-3, (0.9, 0.99), device_type)
    model = torch.compile(raw) if compile_model else raw

    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for i in range(warmup + steps):
        if i == warmup:
            _sync(device)
            t0 = time.time()
        with ctx:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    _sync(device)
    dt = (time.time() - t0) / steps

    tokens = batch_size * cfg.block_size
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
    return {
        "label": label or f"{cfg.n_layer}L/{cfg.n_head}H/{cfg.n_embd}d",
        "params_m": round(raw.num_params() / 1e6, 2),
        "batch_size": batch_size,
        "block_size": cfg.block_size,
        "device": device,
        "dtype": dtype,
        "attention": "flash" if flash else "manual",
        "compiled": compile_model,
        "ms_per_iter": round(dt * 1000, 2),
        "tokens_per_s": round(tokens / dt),
        "mfu_pct": round(raw.estimate_mfu(batch_size, dt) * 100, 2),
        "peak_mem_gb": round(peak_mem, 2),
    }


def print_table(rows: list[dict]) -> None:
    cols = ["label", "params_m", "attention", "compiled", "ms_per_iter", "tokens_per_s",
            "mfu_pct", "peak_mem_gb"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description="Benchmark AdHoc-GPT training throughput")
    p.add_argument("--preset", default="mini", choices=list(PRESETS))
    p.add_argument("--sweep", default=None, choices=["attention", "presets", "batch"],
                   help="attention: flash vs manual | presets: all sizes | batch: batch sweep")
    p.add_argument("--vocab-size", type=int, default=65)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--out", default=None, help="write results as JSON")
    a = p.parse_args(argv)

    def cfg_for(name: str, **kw):
        c = preset_config(name, **kw)
        c.vocab_size = a.vocab_size
        c.dropout = 0.0
        return c

    rows: list[dict] = []
    if a.sweep == "attention":
        for flash in (True, False):
            rows.append(benchmark(cfg_for(a.preset), a.batch_size, a.steps, a.warmup,
                                  a.device, a.dtype, flash, a.compile,
                                  label=f"{a.preset} ({'flash' if flash else 'manual'})"))
    elif a.sweep == "presets":
        for name in PRESETS:
            try:
                rows.append(benchmark(cfg_for(name), a.batch_size, a.steps, a.warmup,
                                      a.device, a.dtype, True, a.compile, label=name))
            except torch.cuda.OutOfMemoryError:  # pragma: no cover
                print(f"{name}: out of memory at batch_size={a.batch_size}")
                torch.cuda.empty_cache()
    elif a.sweep == "batch":
        bs = a.batch_size
        while bs >= 1:
            try:
                rows.append(benchmark(cfg_for(a.preset), bs, a.steps, a.warmup,
                                      a.device, a.dtype, True, a.compile,
                                      label=f"{a.preset} bs={bs}"))
            except torch.cuda.OutOfMemoryError:  # pragma: no cover
                print(f"batch_size={bs}: out of memory")
                torch.cuda.empty_cache()
            bs //= 2
    else:
        rows.append(benchmark(cfg_for(a.preset), a.batch_size, a.steps, a.warmup,
                              a.device, a.dtype, True, a.compile, label=a.preset))

    print_table(rows)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {a.out}")
    return rows


if __name__ == "__main__":
    main()
