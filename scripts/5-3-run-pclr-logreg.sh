#!/usr/bin/env bash
#   slurm/submit.sh --gres=gpu:1 -- scripts/5-3-run-pclr-logreg.sh <dataset> <run> <seed>
# see scripts/_common.sh for the arguments and where outputs go
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

# experiment parameters
EXP_NAME="pclr-logreg"
PRETRAIN_RUN=$PRETRAIN_DIR/pclr-heedb

stage "$EXP_NAME"

python ecg-fms/_compute_pclr_embeddings.py \
--dataset-path $DATASET_PATH \
--ckpt $PRETRAIN_RUN/train-contraster/latest/best.ckpt \
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
