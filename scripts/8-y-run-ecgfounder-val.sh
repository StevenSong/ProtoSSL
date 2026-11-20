#!/usr/bin/env bash
#   slurm/submit.sh --gres=gpu:1 -w kg35-nvl01 -- scripts/8-y-run-ecgfounder-val.sh <dataset> <run> <seed>
# see scripts/_common.sh for the arguments and where outputs go
#
# ECGFounder embeddings of the val split, for the exploratory known-findings adjustment
# in results/proto-recur-nonclassical.ipynb (train/test come from 5-1's
# ecgfounder-logreg). The embeddings don't depend on <seed>, it only places the output.
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

# experiment parameters
EXP_NAME="ecgfounder-val"

stage "$EXP_NAME"

verify_sha256 "$ECGFOUNDER_CKPT" "$ECGFOUNDER_SHA256"
python ecg-fms/_compute_ecgfounder_embeddings.py \
--ckpt $ECGFOUNDER_CKPT \
--dataset-path $DATASET_PATH \
--output-path $RUN_DIR/$EXP_NAME \
--splits val

publish "$EXP_NAME"
