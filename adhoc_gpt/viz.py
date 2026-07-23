"""Phase 5: visualisation -- look inside the model, not just at its loss curve.

    python -m adhoc_gpt.viz attention   --ckpt runs/mini-adhoc-lm/ckpt.pt --text "ROMEO:"
    python -m adhoc_gpt.viz token-loss  --ckpt runs/mini-adhoc-lm/ckpt.pt --text "..."
    python -m adhoc_gpt.viz embeddings  --ckpt runs/mini-adhoc-lm/ckpt.pt
    python -m adhoc_gpt.viz compare     --runs runs/mini-adhoc-lm runs/mini-bpe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .generate import load_model  # noqa: E402
from .plots import read_metrics  # noqa: E402


def _labels(tokenizer, ids: list[int]) -> list[str]:
    out = []
    for i in ids:
        s = tokenizer.decode([i])
        out.append({"\n": "\\n", " ": "␣", "\t": "\\t"}.get(s, s) or "?")
    return out


@torch.no_grad()
def attention_maps(
    ckpt: str | Path, text: str, out: str | Path | None = None, layer: int | None = None,
    device: str = "auto",
) -> Path:
    """Heat-map of every head's attention pattern for one layer."""
    model, tok, device = load_model(ckpt, device)
    model.set_flash(False)  # the manual path is the one we can instrument

    ids = tok.encode(text)[: model.config.block_size]
    if len(ids) < 2:
        raise SystemExit("give at least two tokens of text")
    idx = torch.tensor(ids, device=device)[None, ...]

    captured: dict[int, torch.Tensor] = {}

    def hook(i):
        def fn(module, args, kwargs, output):
            x = args[0]
            B, T, C = x.shape
            q, k, _ = module.c_attn(x).split(module.n_embd, dim=2)
            q = q.view(B, T, module.n_head, module.head_dim).transpose(1, 2)
            k = k.view(B, T, module.n_head, module.head_dim).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / module.head_dim**0.5
            att = att.masked_fill(module.mask[:, :, :T, :T] == 0, float("-inf"))
            captured[i] = F.softmax(att, dim=-1)[0].cpu()
        return fn

    handles = [
        block.attn.register_forward_hook(hook(i), with_kwargs=True)
        for i, block in enumerate(model.transformer.h)
    ]
    model(idx)
    for h in handles:
        h.remove()

    layers = [layer] if layer is not None else sorted(captured)
    n_head = model.config.n_head
    labels = _labels(tok, ids)
    fig, axes = plt.subplots(
        len(layers), n_head, figsize=(2.1 * n_head, 2.3 * len(layers)), squeeze=False
    )
    for r, li in enumerate(layers):
        for h in range(n_head):
            ax = axes[r][h]
            ax.imshow(captured[li][h], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"L{li}H{h}", fontsize=8)
            if len(ids) <= 32:
                ax.set_xticks(range(len(ids)), labels, fontsize=5, rotation=90)
                ax.set_yticks(range(len(ids)), labels, fontsize=5)
            else:
                ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"Attention (rows = query position, cols = key position)\n{text[:60]!r}",
                 fontsize=10)
    fig.tight_layout()
    out = Path(out or Path(ckpt).parent / "attention.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return out


@torch.no_grad()
def token_loss(ckpt: str | Path, text: str, out: str | Path | None = None,
               device: str = "auto") -> Path:
    """Per-token loss / predicted-probability strip -- where the model is surprised."""
    model, tok, device = load_model(ckpt, device)
    ids = tok.encode(text)[: model.config.block_size]
    if len(ids) < 3:
        raise SystemExit("give at least three tokens of text")
    idx = torch.tensor(ids, device=device)[None, ...]

    x = model.transformer.drop(
        model.transformer.wte(idx) + model.transformer.wpe(torch.arange(len(ids), device=device))
    )
    for block in model.transformer.h:
        x, _ = block(x)
    logits = model.lm_head(model.transformer.ln_f(x))[0]
    losses = F.cross_entropy(logits[:-1], idx[0, 1:], reduction="none").cpu()
    probs = F.softmax(logits[:-1], dim=-1).gather(1, idx[0, 1:, None]).squeeze(1).cpu()

    labels = _labels(tok, ids[1:])
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(labels) * 0.28), 5), sharex=True)
    axes[0].bar(range(len(losses)), losses, color="tab:red")
    axes[0].set_ylabel("loss (nats)")
    axes[0].set_title(f"Per-token loss — mean {losses.mean():.3f}, "
                      f"perplexity {losses.mean().exp():.2f}")
    axes[1].bar(range(len(probs)), probs, color="tab:blue")
    axes[1].set_ylabel("P(actual token)")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(len(labels)), labels, fontsize=6, rotation=90)
    for ax in axes:
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = Path(out or Path(ckpt).parent / "token_loss.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return out


@torch.no_grad()
def embedding_projection(ckpt: str | Path, out: str | Path | None = None, top: int = 200,
                         device: str = "auto") -> Path:
    """2-D PCA of the learned token embeddings (PCA computed with plain SVD)."""
    model, tok, _ = load_model(ckpt, device)
    W = model.transformer.wte.weight.detach().cpu()[:top]
    W = W - W.mean(0, keepdim=True)
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    xy = (U[:, :2] * S[:2])
    var = (S**2 / (S**2).sum())[:2] * 100

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.35, color="tab:purple")
    for i in range(min(top, len(xy))):
        label = tok.decode([i]).strip() or repr(tok.decode([i]))[1:-1]
        ax.annotate(label[:12], (xy[i, 0], xy[i, 1]), fontsize=7, alpha=0.85)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)")
    ax.set_title("Token embedding space (PCA)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = Path(out or Path(ckpt).parent / "embeddings.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def compare_runs(runs: list[str | Path], out: str | Path = "runs/comparison.png",
                 labels: list[str] | None = None) -> Path:
    """Overlay validation curves from several runs (loss vs iteration and vs tokens)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for i, run in enumerate(runs):
        m = read_metrics(run)
        name = (labels[i] if labels and i < len(labels) else Path(run).name)
        axes[0].plot(m["iter"], m["val_loss"], lw=1.8, label=f"{name} (best {min(m['val_loss']):.3f})")
        axes[1].plot([t / 1e6 for t in m["tokens"]], m["val_loss"], lw=1.8, label=name)
    axes[0].set_xlabel("iteration")
    axes[1].set_xlabel("tokens seen (millions)")
    for ax in axes:
        ax.set_ylabel("val loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("Validation loss by iteration")
    axes[1].set_title("Validation loss by tokens seen")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="AdHoc-GPT visualisations")
    sub = p.add_subparsers(dest="cmd", required=True)

    a1 = sub.add_parser("attention", help="attention heat-maps")
    a1.add_argument("--ckpt", required=True)
    a1.add_argument("--text", default="ROMEO: But soft, what light")
    a1.add_argument("--layer", type=int, default=None)
    a1.add_argument("--out", default=None)
    a1.add_argument("--device", default="auto")

    a2 = sub.add_parser("token-loss", help="per-token loss / probability")
    a2.add_argument("--ckpt", required=True)
    a2.add_argument("--text", required=True)
    a2.add_argument("--out", default=None)
    a2.add_argument("--device", default="auto")

    a3 = sub.add_parser("embeddings", help="PCA of token embeddings")
    a3.add_argument("--ckpt", required=True)
    a3.add_argument("--top", type=int, default=200)
    a3.add_argument("--out", default=None)
    a3.add_argument("--device", default="auto")

    a4 = sub.add_parser("compare", help="overlay validation curves from several runs")
    a4.add_argument("--runs", nargs="+", required=True)
    a4.add_argument("--labels", nargs="*", default=None)
    a4.add_argument("--out", default="runs/comparison.png")

    a = p.parse_args(argv)
    if a.cmd == "attention":
        attention_maps(a.ckpt, a.text, a.out, a.layer, a.device)
    elif a.cmd == "token-loss":
        token_loss(a.ckpt, a.text, a.out, a.device)
    elif a.cmd == "embeddings":
        embedding_projection(a.ckpt, a.out, a.top, a.device)
    else:
        compare_runs(a.runs, a.out, a.labels)


if __name__ == "__main__":
    main()
