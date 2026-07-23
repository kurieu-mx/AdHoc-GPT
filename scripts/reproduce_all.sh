#!/usr/bin/env bash
# Reproduce every phase of AdHoc-GPT end to end.
#   bash scripts/reproduce_all.sh            # full run (~1h on one 8GB GPU)
#   FAST=1 bash scripts/reproduce_all.sh     # short run, for smoke-testing the pipeline
set -euo pipefail

PY="${PYTHON:-python}"
if [[ "${FAST:-0}" == "1" ]]; then
  P1_ITERS=200; P2_ITERS=200; P3_ITERS=100; STORIES=2000; DOCS=400
else
  P1_ITERS=5000; P2_ITERS=6000; P3_ITERS=2000; STORIES=60000; DOCS=6000
fi

echo "== Phase 1: core LLM build =============================================="
$PY -m adhoc_gpt.data  --dataset shakespeare --tokenizer char --out-dir data/shakespeare_char
$PY -m adhoc_gpt.train --preset mini --data-dir data/shakespeare_char \
    --out-dir runs/mini-adhoc-lm --max-iters "$P1_ITERS"
$PY -m adhoc_gpt.plots --run runs/mini-adhoc-lm
$PY -m adhoc_gpt.generate --ckpt runs/mini-adhoc-lm/ckpt.pt --prompt "KING RICHARD III:" \
    --tokens 500 --temperature 0.8 --top-k 50 | tee runs/mini-adhoc-lm/sample.txt

echo "== Phase 2: scaling & engineering ======================================="
$PY -m adhoc_gpt.bench --preset mini --sweep attention --out runs/bench_attention.json
$PY -m adhoc_gpt.domain.corpus --out data/raw/diplomacy.txt --docs "$DOCS"
# one BPE vocabulary shared by the pretraining and fine-tuning corpora
$PY -m adhoc_gpt.data --train-tokenizer tinystories data/raw/diplomacy.txt \
    --tokenizer bpe --vocab-size 2048 --out-dir data/shared_vocab
$PY -m adhoc_gpt.data --dataset tinystories --max-docs "$STORIES" \
    --tokenizer-from data/shared_vocab/tokenizer.json --out-dir data/tinystories_bpe
$PY -m adhoc_gpt.train --preset mini --data-dir data/tinystories_bpe \
    --out-dir runs/mini-tinystories --max-iters "$P2_ITERS" --batch-size 48 --dropout 0.0
$PY -m adhoc_gpt.plots --run runs/mini-tinystories

echo "== Phase 3: domain specialization ======================================="
$PY -m adhoc_gpt.data --dataset data/raw/diplomacy.txt \
    --tokenizer-from data/shared_vocab/tokenizer.json --out-dir data/diplomacy_bpe
$PY -m adhoc_gpt.train --data-dir data/diplomacy_bpe --out-dir runs/adhoc-lm-domain \
    --init-from runs/mini-tinystories/ckpt.pt --max-iters "$P3_ITERS" \
    --lr 2e-4 --min-lr 2e-5 --warmup-iters 100 --batch-size 48 --dropout 0.05
$PY -m adhoc_gpt.plots --run runs/adhoc-lm-domain

echo "== Phase 4: application & RAG ==========================================="
$PY -m adhoc_gpt.rag --build --corpus data/raw/diplomacy.txt --library data/clause_library.json
$PY -m adhoc_gpt.app draft --ckpt runs/adhoc-lm-domain/ckpt.pt --topic "climate resilience" \
    --tokens 400 | tee runs/adhoc-lm-domain/draft_climate.txt
echo "(web UI: python -m adhoc_gpt.app serve --ckpt runs/adhoc-lm-domain/ckpt.pt)"

echo "== Phase 5: visualisation ==============================================="
$PY -m adhoc_gpt.viz attention  --ckpt runs/mini-adhoc-lm/ckpt.pt --text "ROMEO: But soft," \
    --out runs/mini-adhoc-lm/attention.png
$PY -m adhoc_gpt.viz token-loss --ckpt runs/adhoc-lm-domain/ckpt.pt \
    --text "Recalling its previous resolutions on climate resilience," \
    --out runs/adhoc-lm-domain/token_loss.png
$PY -m adhoc_gpt.viz embeddings --ckpt runs/mini-adhoc-lm/ckpt.pt \
    --out runs/mini-adhoc-lm/embeddings.png
$PY -m adhoc_gpt.viz compare --runs runs/mini-adhoc-lm runs/mini-tinystories \
    runs/adhoc-lm-domain --out runs/comparison.png

echo "all phases complete"
