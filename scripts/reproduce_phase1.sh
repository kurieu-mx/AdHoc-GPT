#!/usr/bin/env bash
# Reproduce the Phase-1 result end to end: data -> training -> curves -> samples.
# Usage: bash scripts/reproduce_phase1.sh [max_iters]
set -euo pipefail

ITERS="${1:-5000}"
PY="${PYTHON:-python}"
RUN="runs/mini-adhoc-lm"

$PY -m adhoc_gpt.data  --dataset shakespeare --tokenizer char --out-dir data/shakespeare_char
$PY -m adhoc_gpt.train --preset mini --data-dir data/shakespeare_char \
    --out-dir "$RUN" --max-iters "$ITERS" --batch-size 64
$PY -m adhoc_gpt.plots --run "$RUN"
$PY -m adhoc_gpt.generate --ckpt "$RUN/ckpt.pt" --prompt "KING RICHARD III:" \
    --tokens 500 --temperature 0.8 --top-k 50 | tee "$RUN/sample.txt"

echo "done -> $RUN"
