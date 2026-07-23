# 📝 AdHoc‑GPT

*A transformer‑based language model built from scratch in PyTorch, specialized for diplomacy, resolutions, and debate.*

**A GPT trained from zero — architecture, tokenizer, training loop, retrieval and application — with no `nn.MultiheadAttention`, no `nn.LayerNorm`, and no tokenizer library.**

- 🧠 **Transformer from first principles** — causal multi‑head attention, LayerNorm, GELU, KV cache, weight tying, all hand‑written and unit‑tested against `torch.nn` equivalents
- 🔤 **Own byte‑level BPE** — merge training with incremental pair statistics: **1.1 s** for 768 merges over 1.1M characters (a naive recount takes ~6 min), verified against a reference implementation
- 🏋️ **Real training stack** — mixed precision, gradient accumulation, cosine LR with warmup, gradient clipping, checkpoint/resume, memory‑mapped data, DDP via `torchrun`, MFU/throughput benchmarks
- 📉 **Trained models with published curves** — 7.5M‑parameter Mini‑AdHoc‑LM hits **1.4528** val loss on character‑level Shakespeare (nanoGPT baseline ≈1.47)
- 🔎 **Retrieval built from scratch** — BM25 + dense retrieval over the model's own embeddings, reciprocal‑rank fusion, MMR diversity, context‑budgeted prompt assembly
- ✅ **56 tests** — including a gradient check proving no position attends to the future, and KV‑cache/full‑forward equivalence
- ▶️ **One command reproduces everything**: `bash scripts/reproduce_all.sh`

---

## 🌍 Overview

**AdHoc‑GPT** builds a language model from first principles — tokenizer, attention, training loop, everything — and then takes it all the way to an application: a retrieval‑augmented resolution‑drafting tool.

All five phases of the original roadmap are implemented:

| Phase | What it is | Where |
|---|---|---|
| 1. Core LLM Build | tokenizers, embeddings, causal multi‑head attention, feed‑forward, training loop, sampling | `tokenizer.py`, `model.py`, `train.py`, `generate.py` |
| 2. Scaling & Engineering | BPE vocabulary, TinyStories/WikiText, AMP, grad accumulation, `torch.compile`, DDP, throughput benchmarks | `data.py`, `train.py`, `bench.py` |
| 3. Domain Specialization | synthetic diplomacy corpus + fine‑tuning from a pretrained checkpoint | `domain/corpus.py`, `train.py --init-from` |
| 4. Application & RAG | BM25 + embedding retrieval over a clause library, drafting CLI and web UI | `rag.py`, `app.py` |
| 5. Visualization | attention maps, per‑token loss, embedding PCA, run comparison | `viz.py`, `plots.py` |

> **On the domain data:** the diplomacy corpus is *synthetic and templated* — generated in‑repo from clause grammars. The domain model imitates the register of multilateral drafting; it does not reproduce real UN documents or any State's position. See [MODEL_CARD.md](MODEL_CARD.md).

---

## 🚀 Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Phase 1 in three commands
python -m adhoc_gpt.data  --dataset shakespeare --tokenizer char --out-dir data/shakespeare_char
python -m adhoc_gpt.train --preset mini --data-dir data/shakespeare_char --out-dir runs/mini-adhoc-lm
python -m adhoc_gpt.generate --ckpt runs/mini-adhoc-lm/ckpt.pt --prompt "KING RICHARD III:"

# or run every phase end to end (~1h on one 8GB GPU; FAST=1 for a smoke test)
bash scripts/reproduce_all.sh
```

No GPU? `--device cpu --preset nano --max-iters 500` trains a toy model in minutes.

---

## 📊 Results

All numbers below were produced by the code in this repo on a single **RTX 4070 Laptop (8 GB)**.

| run | params | data | tokens seen | best val loss | wall clock |
|---|---|---|---|---|---|
| **Mini‑AdHoc‑LM** (Phase 1) | 7.48M | tiny‑Shakespeare, char vocab 65 | 82M | **1.4528** | 28.9 min |
| **Mini‑TinyStories** (Phase 2) | 8.11M | TinyStories 53.6M chars, BPE vocab 2048 | 74M | **1.5990** | 23.9 min |
| **AdHoc‑LM‑Domain** (Phase 3) | 8.11M | synthetic diplomacy, fine‑tuned from Phase 2 | — | *(training)* | — |

For reference, nanoGPT's character‑level Shakespeare baseline lands at ≈1.47 validation loss — Mini‑AdHoc‑LM reaches **1.4528** with a from‑scratch implementation at 7.5M parameters.

### Training curves

![Mini-AdHoc-LM training curves](runs/mini-adhoc-lm/training_curves.png)

Validation bottoms out at iteration 2500 and then rises while training loss keeps falling — textbook overfitting of a 7.5M‑parameter model on a 1M‑character corpus. The best checkpoint (not the last) is the one saved.

### Sample — Mini‑AdHoc‑LM, prompt `KING RICHARD III:`

```
KING RICHARD III:
And will take it as good as to do thee most her.
A bar to the sigh from here and do I.
O God, let's awhile; I am a crown, and enter,
That love o' the steeds that break it will be dead!
O, what will this at looks may scorn thee,
Or who in the head of deserve, which we would
Have in the second heart, being against my honour
Is lawful back.
```

Trained from random initialisation on 1.1M characters: it has learned speaker headings, blank‑verse line length, punctuation and Early Modern morphology — with no notion of meaning, which is exactly what 7.5M parameters buys.

### Inside the model — attention patterns (layer 0, all 8 heads)

![Attention maps](runs/mini-adhoc-lm/attention_L0.png)

Every map is strictly lower‑triangular: the causal mask holds, so no position can see the future. The heads have specialised — H3 is a pure self/identity head, H1 attends to the previous token, H6 spreads attention diffusely over history, and H0/H5 anchor on the sequence start.

<!--RESULTS_END-->

---

## 🏗️ Phase 1 — the model

Decoder‑only transformer, pre‑norm residual blocks:

```
tokens ─► token embedding + positional embedding ─► dropout
       ─► [ x + CausalSelfAttention(LayerNorm(x))
            x + MLP(LayerNorm(x))              ] × n_layer
       ─► LayerNorm ─► lm_head (weights tied to the token embedding) ─► logits
```

| Component | Where | Notes |
|---|---|---|
| Character tokenizer | `tokenizer.py` | one id per unique character (65 for Shakespeare) |
| Byte‑level BPE | `tokenizer.py` | pair counting + merge table trained from scratch, regex pre‑split, byte fallback so any Unicode round‑trips. Incremental pair statistics: 768 merges over 1.1M characters in **1.1 s** |
| Token + positional embeddings | `model.py` | learned, weight‑tied output head |
| Multi‑head self‑attention | `CausalSelfAttention` | fused QKV projection, `softmax(QKᵀ/√d + causal mask)V` written out by hand; `--no-flash` forces the manual path, the default uses the numerically‑equivalent fused SDPA kernel |
| Causal masking | `CausalSelfAttention.mask` | lower‑triangular buffer, offset‑aware so it stays correct with a KV cache |
| Feed‑forward | `MLP` | C → 4C → GELU → C |
| LayerNorm / GELU | `LayerNorm`, `gelu` | from scratch, unit‑tested against `torch.nn` equivalents |
| KV cache | `AdHocGPT.generate` | O(T) per generated token instead of O(T²) |
| Sampling | `generate.py` | temperature, top‑k, nucleus (top‑p), seeded |

Presets (`config.py`):

| preset | layers | heads | n_embd | context | params (vocab 65) |
|---|---|---|---|---|---|
| `nano` | 4 | 4 | 128 | 128 | 0.8M |
| `mini` | 6 | 8 | 320 | 256 | **7.5M** |
| `small` | 8 | 8 | 512 | 512 | 25M |
| `base` | 12 | 12 | 768 | 1024 | 86M |

---

## ⚙️ Phase 2 — scaling & engineering

```bash
python -m adhoc_gpt.bench --preset mini --sweep attention   # flash vs manual attention
python -m adhoc_gpt.bench --sweep presets --batch-size 32   # ms/iter, tokens/s, MFU, peak memory
python -m adhoc_gpt.bench --sweep batch  --preset small     # find the batch size that fits

# a BPE vocabulary shared by two corpora (required before fine-tuning)
python -m adhoc_gpt.data --train-tokenizer tinystories data/raw/diplomacy.txt \
    --tokenizer bpe --vocab-size 2048 --out-dir data/shared_vocab
python -m adhoc_gpt.data --dataset tinystories --max-docs 60000 \
    --tokenizer-from data/shared_vocab/tokenizer.json --out-dir data/tinystories_bpe

# multi-GPU data parallel
torchrun --standalone --nproc_per_node=4 -m adhoc_gpt.train --preset small \
    --data-dir data/tinystories_bpe --grad-accum-steps 8
```

Also in the loop: mixed precision (bf16/fp16 + GradScaler), gradient accumulation, gradient clipping, cosine LR with warmup, checkpoint/resume, memory‑mapped datasets, `torch.compile`.

---

## 🕊️ Phase 3 — domain specialization

`adhoc_gpt/domain/corpus.py` generates resolutions, verbatim debate records, position papers and procedural records over eight topics. Clauses are **typed by the complement they take** (`np`, `np_to`, `that`, `inf`), so an operative verb is never paired with a phrase it cannot govern:

```
The General Assembly,

Recalling its previous resolutions on climate resilience, in particular resolution A/RES/77/214,

Concerned that the widening adaptation finance gap continues to undermine the objectives of
the Paris Agreement,

1. Urges Member States to allocate predictable and additional resources for national adaptation plans;
2. Decides to establish a voluntary trust fund in support of early warning systems.
```

Fine‑tune a pretrained checkpoint on it (architecture and vocabulary come from the checkpoint):

```bash
python -m adhoc_gpt.data --dataset data/raw/diplomacy.txt \
    --tokenizer-from data/shared_vocab/tokenizer.json --out-dir data/diplomacy_bpe
python -m adhoc_gpt.train --data-dir data/diplomacy_bpe --out-dir runs/adhoc-lm-domain \
    --init-from runs/mini-tinystories/ckpt.pt --lr 2e-4 --max-iters 2000
```

---

## 🔎 Phase 4 — application & RAG

Retrieval is implemented from scratch — no search dependency:

* **BM25** (`rag.py`) — Okapi term weighting over a clause library built from the corpus.
* **Embedding retrieval** — document vectors are mean token embeddings *from the trained model itself*, scored by cosine similarity.
* **Hybrid** — reciprocal‑rank fusion of both, then **MMR** re‑ranking so four retrieved precedents are four *different* clauses instead of four paraphrases.

```bash
python -m adhoc_gpt.rag --build --corpus data/raw/diplomacy.txt --library data/clause_library.json
python -m adhoc_gpt.app draft --topic "maritime security" --tokens 400
python -m adhoc_gpt.app repl                  # interactive drafting
python -m adhoc_gpt.app serve --port 8000     # web UI (stdlib http.server, no framework)
```

Retrieved clauses are formatted into a document header the fine‑tuned model recognises, so it *continues* the resolution rather than describing one. Every draft carries the synthetic‑output disclaimer.

---

## 📈 Phase 5 — visualization

```bash
python -m adhoc_gpt.viz attention  --ckpt runs/mini-adhoc-lm/ckpt.pt --text "ROMEO: But soft,"
python -m adhoc_gpt.viz token-loss --ckpt runs/adhoc-lm-domain/ckpt.pt --text "Recalling its ..."
python -m adhoc_gpt.viz embeddings --ckpt runs/mini-adhoc-lm/ckpt.pt
python -m adhoc_gpt.viz compare    --runs runs/mini-adhoc-lm runs/mini-tinystories
python -m adhoc_gpt.plots --run runs/mini-adhoc-lm      # loss + LR schedule
```

---

## 🧪 Tests

```bash
pytest        # ~2 min on CPU
```

Covers: LayerNorm/GELU parity with `torch.nn`; **no information leaks from future positions** (gradient check); manual vs fused attention agreement; KV‑cache vs full‑forward equivalence; tokenizer round‑trips including Unicode and special tokens; **fast BPE trainer vs a naive recount**; LR schedule; next‑token batch shifting; end‑to‑end training that must reduce the loss; fine‑tuning from a checkpoint (and rejecting a vocabulary mismatch); clause‑grammar type alignment; BM25 ranking, filtering and MMR diversity; the HTTP drafting API; and every plot.

> If your shell has a system `PYTHONPATH` (e.g. ROS), run `env -u PYTHONPATH pytest`.

---

## 🧩 Roadmap

- [x] Initialize repo + environment
- [x] Implement tokenizer + embeddings
- [x] Multi‑head attention
- [x] Feedforward layers, normalization, causal masking
- [x] Training loop for Mini‑AdHoc‑LM
- [x] Pretrain Mini‑AdHoc‑LM on Shakespeare
- [x] Publish initial training curves
- [x] Phase 2 — BPE vocabulary, TinyStories scale‑up, DDP, throughput benchmarks
- [x] Phase 3 — domain corpus + fine‑tuning
- [x] Phase 4 — retrieval‑augmented drafting app (CLI + web UI)
- [x] Phase 5 — attention / embedding / loss visualizations
- [ ] Next: real (non‑synthetic) domain corpora, instruction‑tuning, longer context

---

## 📁 Layout

```
adhoc_gpt/
  config.py       GPTConfig / TrainConfig + model presets
  tokenizer.py    character-level and byte-level BPE tokenizers (from scratch)
  data.py         corpus download, shared-vocab training, tokenization, mmap batching
  model.py        LayerNorm, GELU, CausalSelfAttention, MLP, Block, AdHocGPT
  train.py        training loop (AMP, cosine LR, grad accum, DDP, fine-tuning)
  generate.py     sampling CLI (temperature / top-k / top-p / KV cache)
  bench.py        throughput, MFU and memory benchmarks
  rag.py          BM25 + embedding retrieval, clause library, MMR, prompt assembly
  app.py          drafting application: draft / retrieve / repl / serve
  viz.py          attention maps, per-token loss, embedding PCA, run comparison
  plots.py        training-curve plots
  domain/corpus.py synthetic diplomacy corpus generator
scripts/          reproduce_phase1.sh, reproduce_all.sh
tests/            unit + end-to-end tests
runs/             checkpoints, metrics.csv, curves, samples
```

---

## 🙌 Acknowledgements

- Sebastian Raschka — *“Build a Large Language Model (From Scratch)”*
- Andrej Karpathy — nanoGPT / minBPE, for the tiny‑Shakespeare corpus and the training‑loop shape
- `roneneldan/TinyStories` for the Phase‑2 pretraining corpus
