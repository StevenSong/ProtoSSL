#!/usr/bin/env bash
# HEEDB normalization stats. The repo's copies in data-preprocessing/heedb_normalizations/
# came from this; copy a new run's over them to update.
#   slurm/submit.sh -p h200 -- scripts/pretrain/run-heedb-normalizations.sh <run> <subset>
#   <subset>: mgb, emory or both (see _compute_heedb_normalizations.py)
set -euo pipefail
(($# == 2)) || { echo "usage: $0 <run> <subset>" >&2; exit 1; }
SUBSET=$2
source "$(dirname "$0")/_common.sh" "$1"

EXP_NAME="heedb-normalizations"

stage "$EXP_NAME"

python pretrain/_compute_heedb_normalizations.py --dataset_path $PRETRAIN_DATASET --output_path $RUN_DIR/$EXP_NAME --subset $SUBSET

publish "$EXP_NAME"
