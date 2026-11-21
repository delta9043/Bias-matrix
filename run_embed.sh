#!/usr/bin/env bash
# Embedding extraction for all splits (resumable: finished chunks are skipped).
# Usage: bash run_embed.sh   (from the repository root; PYTHON=<interpreter> to pick the environment)
set -e
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
PY="${PYTHON:-python}"
DATA="${BIAS_MATRIX_DATA:-data}"

# Bias in Bios: original text (train / dev / test) and gender-swapped training text (CDA)
$PY -m experiments.embed --split test
$PY -m experiments.embed --split train
$PY -m experiments.embed --split dev
$PY -m experiments.embed --split train --kind cf

# Korean Public Data: hazard consultations (CSV placed by hand, see README); long case bodies -> 256 tokens
if [ -f "$DATA/consumer/hazard_15098320.csv" ]; then
  $PY -m experiments.embed --split train --kind hazard --max-len 256 --batch 64 --chunk 4096
  $PY -m experiments.embed --split test --kind hazard --max-len 256 --batch 64 --chunk 4096
fi
