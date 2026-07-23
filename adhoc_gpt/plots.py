"""Plot training curves from a run's ``metrics.csv``.

    python -m adhoc_gpt.plots --run runs/mini-adhoc-lm
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402


def read_metrics(run_dir: str | Path) -> dict[str, list[float]]:
    path = Path(run_dir) / "metrics.csv"
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"{path} is empty -- has training run?")
    cols: dict[str, list[float]] = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            cols[k].append(float(v))
    return cols


def plot_run(run_dir: str | Path, out: str | Path | None = None, title: str | None = None) -> Path:
    run_dir = Path(run_dir)
    m = read_metrics(run_dir)
    out = Path(out or run_dir / "training_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(m["iter"], m["train_loss"], label="train", lw=1.8)
    ax.plot(m["iter"], m["val_loss"], label="val", lw=1.8)
    best = min(m["val_loss"])
    best_it = m["iter"][m["val_loss"].index(best)]
    ax.scatter([best_it], [best], zorder=5, s=28, color="crimson")
    ax.annotate(f"best val {best:.4f}", (best_it, best), textcoords="offset points",
                xytext=(6, 10), fontsize=9, color="crimson")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title or f"{run_dir.name}: loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(m["iter"], m["lr"], color="tab:green", lw=1.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("learning rate")
    ax.set_title("LR schedule (warmup + cosine decay)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Plot AdHoc-GPT training curves")
    p.add_argument("--run", default="runs/mini-adhoc-lm")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    a = p.parse_args(argv)
    plot_run(a.run, a.out, a.title)


if __name__ == "__main__":
    main()
