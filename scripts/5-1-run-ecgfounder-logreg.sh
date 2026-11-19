#!/usr/bin/env bash
#   slurm/submit.sh --gres=gpu:1 -- scripts/5-1-run-ecgfounder-logreg.sh <dataset> <run> <seed>
# see scripts/_common.sh for the arguments and where outputs go
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

# experiment parameters
EXP_NAME="ecgfounder-logreg"

stage "$EXP_NAME"

verify_sha256 "$ECGFOUNDER_CKPT" "$ECGFOUNDER_SHA256"
python ecg-fms/_compute_ecgfounder_embeddings.py \
--ckpt $ECGFOUNDER_CKPT \
--dataset-path $DATASET_PATH \
--output-path $RUN_DIR/$EXP_NAME

python _linear_probe.py \
--random-seed $SEED \
--dataset-path $DATASET_PATH \
--prototype-embeddings $RUN_DIR/$EXP_NAME \
--output-path $RUN_DIR/$EXP_NAME

python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

python _eval_probs_bootstrapped.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

publish "$EXP_NAME"
