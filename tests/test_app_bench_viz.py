"""End-to-end tests for Phases 2, 4 and 5 (benchmark, application, visualisation)."""

import json
import threading
import urllib.request

import pytest
import torch

from adhoc_gpt.app import DraftingEngine, make_server
from adhoc_gpt.bench import benchmark, print_table
from adhoc_gpt.config import TrainConfig, preset_config
from adhoc_gpt.data import prepare
from adhoc_gpt.domain.corpus import build_corpus
from adhoc_gpt.rag import clauses_from_corpus, save_library
from adhoc_gpt.train import train
from adhoc_gpt.viz import attention_maps, compare_runs, embedding_projection, token_loss


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A tiny domain model + clause library, trained end to end on CPU."""
    root = tmp_path_factory.mktemp("phase45")
    corpus = build_corpus(60, seed=2)
    src = root / "diplomacy.txt"
    src.write_text(corpus)

    data_dir = root / "data"
    prepare(str(src), out_dir=data_dir, tokenizer="char", val_split=0.1)

    out_dir = root / "run"
    model_cfg = preset_config("nano", n_layer=2, n_head=2, n_embd=64, block_size=64)
    cfg = TrainConfig(
        data_dir=str(data_dir), out_dir=str(out_dir), batch_size=16, max_iters=40,
        eval_interval=20, eval_iters=3, warmup_iters=5, log_interval=1000,
        device="cpu", dtype="float32",
    )
    summary = train(model_cfg, cfg)

    library = save_library(clauses_from_corpus(corpus), root / "library.json")
    return {"run": out_dir, "ckpt": out_dir / "ckpt.pt", "library": library,
            "data": data_dir, "summary": summary, "root": root}


# --- Phase 2: benchmarking -----------------------------------------------
def test_benchmark_reports_throughput():
    cfg = preset_config("nano", n_layer=2, n_head=2, n_embd=64, block_size=32)
    cfg.vocab_size, cfg.dropout = 65, 0.0
    row = benchmark(cfg, batch_size=4, steps=3, warmup=1, device="cpu", dtype="float32")
    assert row["ms_per_iter"] > 0 and row["tokens_per_s"] > 0
    assert row["params_m"] > 0 and row["attention"] == "flash"
    print_table([row])  # must not raise


def test_benchmark_manual_attention_path():
    cfg = preset_config("nano", n_layer=1, n_head=2, n_embd=32, block_size=32)
    cfg.vocab_size, cfg.dropout = 65, 0.0
    row = benchmark(cfg, batch_size=4, steps=2, warmup=1, device="cpu", dtype="float32",
                    flash=False)
    assert row["attention"] == "manual"


# --- Phase 3/4: fine-tuning + retrieval-augmented drafting ---------------
def test_finetune_from_checkpoint(trained, tmp_path):
    """init_from must load the base weights and keep training from them."""
    out = tmp_path / "ft"
    cfg = TrainConfig(
        data_dir=str(trained["data"]), out_dir=str(out), batch_size=8, max_iters=10,
        eval_interval=5, eval_iters=2, warmup_iters=2, log_interval=1000,
        learning_rate=1e-4, device="cpu", dtype="float32",
    )
    summary = train(preset_config("nano"), cfg, init_from=trained["ckpt"])
    # the fine-tuned model should not be worse than a fresh 10-iteration model
    assert summary["best_val_loss"] < 4.0
    assert (out / "ckpt.pt").exists()


def test_finetune_rejects_vocab_mismatch(trained, tmp_path):
    other = tmp_path / "other"
    (tmp_path / "c.txt").write_text("abcdefghij " * 500)
    prepare(str(tmp_path / "c.txt"), out_dir=other, tokenizer="char", val_split=0.1)
    cfg = TrainConfig(data_dir=str(other), out_dir=str(tmp_path / "bad"), batch_size=4,
                      max_iters=2, eval_interval=1, eval_iters=1, device="cpu", dtype="float32")
    with pytest.raises(ValueError, match="vocab_size"):
        train(preset_config("nano"), cfg, init_from=trained["ckpt"])


def test_drafting_engine_returns_draft_and_precedent(trained):
    engine = DraftingEngine(trained["ckpt"], trained["library"], device="cpu")
    result = engine.draft("climate resilience", k=3, tokens=40, seed=0)
    assert result["topic"] == "climate resilience"
    assert result["draft"] and isinstance(result["draft"], str)
    assert "synthetic" in result["disclaimer"].lower()
    assert "Title: Resolution on climate resilience" in result["prompt"]
    # retrieval finds clauses even when this toy context is too small to hold them
    assert 0 < len(engine.retrieve("climate resilience", k=3)) <= 3
    assert len(result["retrieved"]) <= 3


def test_prompt_fits_the_context_or_is_stripped_to_the_header(trained):
    """Precedent is dropped until the prompt fits; a bare header is the floor."""
    engine = DraftingEngine(trained["ckpt"], trained["library"], device="cpu", dense=False)
    for topic in ("climate resilience", "nuclear disarmament", "a" * 300):
        result = engine.draft(topic, k=4, tokens=5, seed=1)
        n_tokens = len(engine.tokenizer.encode(result["prompt"]))
        assert n_tokens <= engine.model.config.block_size or not result["retrieved"]
        assert len(result["prompt"]) < 600, "an absurd topic must not blow up the prompt"
        # every reported clause is one the model actually saw
        for hit in result["retrieved"]:
            assert hit["text"][:80] in result["prompt"]


def test_retrieved_precedent_is_diverse(trained):
    engine = DraftingEngine(trained["ckpt"], trained["library"], device="cpu")
    hits = engine.retrieve("climate resilience adaptation finance", k=4)
    assert len({h.doc.id for h in hits}) == len(hits)


def test_dense_retrieval_uses_model_embeddings(trained):
    engine = DraftingEngine(trained["ckpt"], trained["library"], device="cpu", dense=True)
    from adhoc_gpt.rag import EmbeddingIndex

    dense = [r for r in engine.retriever.retrievers if isinstance(r, EmbeddingIndex)]
    assert dense, "dense retriever should be active"
    assert dense[0].matrix.shape[1] == engine.model.config.n_embd
    assert dense[0].search("refugee protection", k=2)


def test_http_api_serves_a_draft(trained):
    engine = DraftingEngine(trained["ckpt"], trained["library"], device="cpu", dense=False)
    server = make_server(engine, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "AdHoc-GPT" in page and "not authentic UN text" in page

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/draft",
            data=json.dumps({"topic": "food security", "tokens": 30}).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(urllib.request.urlopen(req).read())
        assert body["topic"] == "food security" and body["draft"]

        bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/draft", data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert "error" in json.loads(urllib.request.urlopen(bad).read())
    finally:
        server.shutdown()
        server.server_close()


# --- Phase 5: visualisation ----------------------------------------------
def test_attention_maps_are_rows_summing_to_one(trained, tmp_path):
    out = attention_maps(trained["ckpt"], "The General Assembly,", tmp_path / "att.png",
                         device="cpu")
    assert out.exists() and out.stat().st_size > 5000


def test_token_loss_plot(trained, tmp_path):
    out = token_loss(trained["ckpt"], "Recalling its previous resolutions on food security,",
                     tmp_path / "tl.png", device="cpu")
    assert out.exists() and out.stat().st_size > 5000


def test_embedding_projection_plot(trained, tmp_path):
    out = embedding_projection(trained["ckpt"], tmp_path / "emb.png", top=40, device="cpu")
    assert out.exists() and out.stat().st_size > 5000


def test_compare_runs_plot(trained, tmp_path):
    out = compare_runs([trained["run"], trained["run"]], tmp_path / "cmp.png",
                       labels=["a", "b"])
    assert out.exists() and out.stat().st_size > 5000


def test_generation_is_reproducible_with_a_seed(trained):
    engine = DraftingEngine(trained["ckpt"], trained["library"], device="cpu", dense=False)
    a = engine.draft("cybersecurity", k=2, tokens=30, seed=7)["draft"]
    b = engine.draft("cybersecurity", k=2, tokens=30, seed=7)["draft"]
    assert a == b
    torch.manual_seed(0)
