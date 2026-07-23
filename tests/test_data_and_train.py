import json

import pytest
import torch

from adhoc_gpt.config import TrainConfig, preset_config
from adhoc_gpt.data import BinDataset, prepare
from adhoc_gpt.plots import plot_run
from adhoc_gpt.train import get_lr, train

CORPUS = (
    "ROMEO:\nBut soft, what light through yonder window breaks?\n"
    "JULIET:\nO Romeo, Romeo, wherefore art thou Romeo?\n"
) * 200


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    src = d / "corpus.txt"
    src.write_text(CORPUS)
    out = d / "prepared"
    prepare(str(src), out_dir=out, tokenizer="char", val_split=0.1)
    return out


def test_prepare_writes_bins_and_meta(data_dir):
    meta = json.loads((data_dir / "meta.json").read_text())
    assert meta["tokenizer"] == "char"
    assert meta["train_tokens"] > 0 and meta["val_tokens"] > 0
    assert (data_dir / "train.bin").exists() and (data_dir / "tokenizer.json").exists()
    assert meta["vocab_size"] == len(set(CORPUS))


def test_batches_are_next_token_shifted(data_dir):
    ds = BinDataset(data_dir, "train", block_size=16)
    x, y = ds.get_batch(8)
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == torch.int64
    # y is x shifted by one token
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_dataset_rejects_too_small_block(data_dir):
    with pytest.raises(ValueError):
        BinDataset(data_dir, "val", block_size=10**6)


def test_lr_schedule_warms_up_then_decays():
    cfg = TrainConfig(learning_rate=1e-3, min_lr=1e-4, warmup_iters=10, max_iters=100)
    assert get_lr(0, cfg) == pytest.approx(1e-4)          # first warmup step
    assert get_lr(9, cfg) == pytest.approx(1e-3)          # peak at end of warmup
    assert get_lr(100, cfg) == pytest.approx(1e-4)        # decayed to min_lr
    assert get_lr(50, cfg) < get_lr(20, cfg)              # monotone decay after warmup


def test_end_to_end_training_reduces_loss(data_dir, tmp_path):
    model_cfg = preset_config("nano", n_layer=2, n_head=2, n_embd=32, block_size=32)
    cfg = TrainConfig(
        data_dir=str(data_dir), out_dir=str(tmp_path / "run"),
        batch_size=16, max_iters=60, eval_interval=20, eval_iters=5,
        warmup_iters=5, log_interval=1000, device="cpu", dtype="float32",
    )
    summary = train(model_cfg, cfg)

    out = tmp_path / "run"
    assert (out / "ckpt.pt").exists() and (out / "last.pt").exists()
    assert (out / "metrics.csv").exists()
    # started near ln(vocab) and improved
    assert summary["best_val_loss"] < 3.0
    plot_path = plot_run(out)
    assert plot_path.exists() and plot_path.stat().st_size > 1000


def test_checkpoint_reloads_into_a_working_model(tmp_path, data_dir):
    from adhoc_gpt.generate import load_model

    out = tmp_path / "run2"
    model_cfg = preset_config("nano", n_layer=1, n_head=2, n_embd=32, block_size=32)
    cfg = TrainConfig(
        data_dir=str(data_dir), out_dir=str(out), batch_size=8, max_iters=10,
        eval_interval=5, eval_iters=2, warmup_iters=2, log_interval=1000,
        device="cpu", dtype="float32",
    )
    train(model_cfg, cfg)
    model, tok, device = load_model(out / "ckpt.pt", device="cpu")
    text = tok.decode(
        model.generate(torch.zeros((1, 1), dtype=torch.long), 20, top_k=5)[0].tolist()
    )
    assert len(text) == 21
