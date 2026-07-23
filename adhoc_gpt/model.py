"""The AdHoc-GPT decoder-only transformer, written from first principles.

Everything below is built out of ``nn.Linear``/``nn.Embedding`` and raw tensor
maths -- no ``nn.MultiheadAttention``, no ``nn.TransformerEncoder``,
no ``nn.LayerNorm``. The one optional shortcut is PyTorch's fused
scaled-dot-product-attention kernel, which is numerically equivalent to the
manual path in :meth:`CausalSelfAttention._manual_attention` and can be turned
off with ``config.flash = False`` / ``model.set_flash(False)``.
"""

from __future__ import annotations

import inspect
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit (tanh approximation used by GPT-2)."""
    return (
        0.5
        * x
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


class LayerNorm(nn.Module):
    """Layer normalisation with an optional bias (PyTorch's has no on/off switch)."""

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        xhat = (x - mean) * torch.rsqrt(var + self.eps)
        out = xhat * self.weight
        return out if self.bias is None else out + self.bias


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (autoregressive) mask.

    Shapes, with B=batch, T=time, C=n_embd, nh=n_head, hd=head_dim:
    ``x (B,T,C) -> q,k,v (B,nh,T,hd) -> att (B,nh,T,T) -> y (B,T,C)``
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        # one fused projection for q, k and v
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        # lower-triangular mask, registered as a buffer so it moves with .to(device)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_offset: int
    ) -> torch.Tensor:
        T_q, T_k = q.size(-2), k.size(-2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # a query at absolute position q_offset+i may attend to keys 0..q_offset+i
        causal = self.mask[:, :, q_offset : q_offset + T_q, :T_k]
        att = att.masked_fill(causal == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q_offset = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            q_offset = past_k.size(-2)
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present = (k, v) if use_cache else None

        if self.flash:
            # with a cache the queries are the *last* T positions, which is what
            # is_causal expects only when T_q == T_k; otherwise pass the mask.
            if q_offset == 0:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                attn_mask = self.mask[:, :, q_offset : q_offset + T, : k.size(-2)] == 1
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                )
        else:
            y = self._manual_attention(q, k, v, q_offset)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads
        y = self.resid_dropout(self.c_proj(y))
        return y, present


class MLP(nn.Module):
    """Position-wise feed-forward network: C -> 4C -> GELU -> C."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache: bool = False):
        attn_out, present = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present


class AdHocGPT(nn.Module):
    """A GPT-style decoder-only language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),   # position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 §2.3)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # --- introspection ---------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
            if not self.config.tie_weights:
                n -= self.transformer.wte.weight.numel()
        return n

    def set_flash(self, enabled: bool) -> None:
        """Toggle the fused attention kernel (falls back to the manual maths)."""
        for block in self.transformer.h:
            block.attn.flash = enabled and hasattr(F, "scaled_dot_product_attention")

    # --- forward ---------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kvs: list | None = None,
        use_cache: bool = False,
    ):
        device = idx.device
        b, t = idx.size()
        past_len = past_kvs[0][0].size(-2) if past_kvs else 0
        if past_len + t > self.config.block_size:
            raise ValueError(
                f"sequence of length {past_len + t} exceeds block_size {self.config.block_size}"
            )
        pos = torch.arange(past_len, past_len + t, dtype=torch.long, device=device)

        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        presents = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past = past_kvs[i] if past_kvs is not None else None
            x, present = block(x, past_kv=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        else:
            # inference shortcut: only the last position feeds the next token
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return (logits, loss, presents) if use_cache else (logits, loss)

    # --- optimisation ----------------------------------------------------
    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str
    ) -> torch.optim.AdamW:
        """AdamW with decay on matmul weights only (not biases / norms / embeddings-1D)."""
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in params.values() if p.dim() >= 2]
        no_decay = [p for p in params.values() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = {"fused": True} if (fused_ok and device_type == "cuda") else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float, flops_promised: float = 3.5e13) -> float:
        """Model-flops-utilisation estimate (PaLM appendix B); default peak is a 4070 laptop bf16."""
        n = self.num_params()
        cfg = self.config
        flops_per_token = 6 * n + 12 * cfg.n_layer * cfg.n_head * cfg.head_dim * cfg.block_size
        flops_per_iter = flops_per_token * cfg.block_size * fwdbwd_per_iter
        return (flops_per_iter / dt) / flops_promised

    # --- generation ------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
        eos_token: int | None = None,
        suppress_tokens: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` continuations of ``idx``.

        ``top_k`` keeps the k most likely tokens; ``top_p`` keeps the smallest
        set whose cumulative probability exceeds p (nucleus sampling). Both may
        be combined. A KV cache makes each step O(T) instead of O(T^2).

        ``suppress_tokens`` bans token ids outright -- used to stop a model from
        emitting a document separator before it has written anything.
        """
        banned = (
            torch.tensor(list(suppress_tokens), dtype=torch.long, device=idx.device)
            if suppress_tokens else None
        )
        was_training = self.training
        self.eval()
        past_kvs = None
        cur = idx
        for _ in range(max_new_tokens):
            # crop the context to block_size (cache path only feeds the new token)
            if past_kvs is None:
                cur_in = cur[:, -self.config.block_size :]
            else:
                if past_kvs[0][0].size(-2) >= self.config.block_size:
                    past_kvs = None  # cache full: recompute from the cropped window
                    cur_in = cur[:, -self.config.block_size + 1 :]
                else:
                    cur_in = cur[:, -1:]

            if use_cache:
                logits, _, past_kvs = self(cur_in, past_kvs=past_kvs, use_cache=True)
            else:
                logits, _ = self(cur[:, -self.config.block_size :])

            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if banned is not None:
                logits.index_fill_(1, banned, float("-inf"))

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < thresh, float("-inf"))

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = probs - F.softmax(sorted_logits, dim=-1) >= top_p
                remove[:, 0] = False  # always keep the top token
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            cur = torch.cat((cur, next_id), dim=1)
            if eos_token is not None and bool((next_id == eos_token).all()):
                break

        if was_training:
            self.train()
        return cur
