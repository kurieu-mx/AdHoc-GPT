import pytest
import torch

from adhoc_gpt.config import GPTConfig, preset_config
from adhoc_gpt.model import AdHocGPT, LayerNorm, gelu

torch.manual_seed(0)


def tiny_config(**kw) -> GPTConfig:
    base = dict(vocab_size=41, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_layernorm_matches_torch():
    x = torch.randn(4, 7, 32)
    mine = LayerNorm(32, bias=True)
    ref = torch.nn.LayerNorm(32)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_gelu_matches_torch():
    x = torch.randn(1000)
    assert torch.allclose(gelu(x), torch.nn.functional.gelu(x, approximate="tanh"), atol=1e-6)


def test_forward_shapes_and_loss():
    cfg = tiny_config()
    model = AdHocGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(float(cfg.vocab_size))), rel=0.15)
    # without targets only the last position is projected (inference shortcut)
    logits, loss = model(idx)
    assert logits.shape == (3, 1, cfg.vocab_size) and loss is None


def test_block_size_is_enforced():
    cfg = tiny_config()
    model = AdHocGPT(cfg)
    with pytest.raises(ValueError):
        model(torch.zeros((1, cfg.block_size + 1), dtype=torch.long))


def test_attention_is_causal():
    """Position t's logits must not depend on any input after t."""
    cfg = tiny_config()
    model = AdHocGPT(cfg).eval()
    emb = torch.randn(1, cfg.block_size, cfg.n_embd, requires_grad=True)

    x = model.transformer.drop(emb + model.transformer.wpe(torch.arange(cfg.block_size)))
    for block in model.transformer.h:
        x, _ = block(x)
    out = model.lm_head(model.transformer.ln_f(x))

    t = 5
    out[0, t].sum().backward()
    grad = emb.grad[0]
    assert grad[: t + 1].abs().sum() > 0          # depends on the past
    assert grad[t + 1 :].abs().max().item() == 0  # never on the future


def test_manual_and_flash_attention_agree():
    cfg = tiny_config()
    model = AdHocGPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    model.set_flash(True)
    flash_logits, _ = model(idx, idx)
    model.set_flash(False)
    manual_logits, _ = model(idx, idx)
    assert torch.allclose(flash_logits, manual_logits, atol=1e-4)


def test_kv_cache_matches_full_forward():
    cfg = tiny_config()
    model = AdHocGPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))

    full, _ = model(idx)                   # logits for the final position
    past, out = None, None
    for t in range(idx.size(1)):
        out, _, past = model(idx[:, t : t + 1], past_kvs=past, use_cache=True)
    assert torch.allclose(full[:, -1], out[:, -1], atol=1e-4)


def test_generate_shapes_and_sampling_modes():
    cfg = tiny_config()
    model = AdHocGPT(cfg).eval()
    idx = torch.zeros((2, 1), dtype=torch.long)
    for kwargs in ({"top_k": 3}, {"top_p": 0.9}, {"top_k": 5, "top_p": 0.95}, {"use_cache": False}):
        out = model.generate(idx, max_new_tokens=10, **kwargs)
        assert out.shape == (2, 11)
        assert out.min() >= 0 and out.max() < cfg.vocab_size


def test_generate_past_block_size():
    """Generation must keep working once the context window is full."""
    cfg = tiny_config(block_size=8)
    model = AdHocGPT(cfg).eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    assert out.shape == (1, 21)


def test_weight_tying_and_param_count():
    cfg = tiny_config()
    model = AdHocGPT(cfg)
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
    untied = AdHocGPT(tiny_config(tie_weights=False))
    assert untied.num_params() > model.num_params()


def test_mini_preset_is_5_to_10M_params():
    cfg = preset_config("mini", **{"n_layer": None})
    cfg.vocab_size = 65
    n = AdHocGPT(cfg).num_params()
    assert 5e6 <= n <= 10e6, f"mini preset has {n/1e6:.2f}M params"


def test_optimizer_groups_split_decay():
    model = AdHocGPT(tiny_config(bias=True))
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cpu")
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_can_overfit_a_single_batch():
    """Sanity check that the gradients actually train the model."""
    torch.manual_seed(0)
    cfg = tiny_config(dropout=0.0)
    model = AdHocGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(120):
        _, loss = model(idx, idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.2 * losses[0], f"{losses[0]:.3f} -> {losses[-1]:.3f}"
