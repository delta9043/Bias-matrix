#!/usr/bin/env bash
# All experiments of the paper after embedding extraction, then figures and tables.
# Usage: bash run_experiments.sh   (from the repository root; PYTHON=<interpreter> to pick the environment)
# Results go to results/ (the shipped files are overwritten); BIAS_MATRIX_RESULTS=<dir> writes elsewhere.
set -e
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
PY="${PYTHON:-python}"
DATA="${BIAS_MATRIX_DATA:-data}"

$PY -m pytest tests -q
$PY -m experiments.select_layer            # layer and L2 by dev accuracy -> select.json
$PY -m experiments.spectrum                # bias spectrum, k of the permutation test (Figure 1)
$PY -m experiments.removal                 # embedding edits (Table 2 upper block, Figure 2b-c)
$PY -m experiments.frozen                  # BME without retraining (Table 2)
$PY -m experiments.augmentation            # reweighing, BMA, CDA (Table 2 lower block, Figures 2b, 3c)
$PY -m experiments.dose --data bios        # bias strength of the training data (Figures 2a, 3a-b)
$PY -m experiments.tradeoff --data bios    # accuracy vs. gap by the number of directions (Figure 2b)

# Korean Public Data (embeddings extracted separately, see README)
if [ -d "$DATA/emb/Qwen3-0.6B/hazard_train" ]; then
  $PY -m experiments.hazard                # two groups (Table 3 upper block, Figure 4b-c)
  $PY -m experiments.multigroup            # three groups (Table 3 lower block)
  $PY -m experiments.rare_cells            # recall of the rare cells (Section 5.4)
  $PY -m experiments.dose --data hazard    # Figure 4a
  $PY -m experiments.tradeoff --data hazard --inlp   # Figure 4b
fi

$PY -m experiments.figures
$PY -m experiments.tables
