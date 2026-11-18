#!/usr/bin/env bash
#   slurm/submit.sh --gres=gpu:1 -- scripts/5-2-run-stmem-logreg.sh <dataset> <run> <seed>
# see scripts/_common.sh for the arguments and where outputs go
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

# experiment parameters
EXP_NAME="stmem-logreg"

stage "$EXP_NAME"

verify_sha256 "$STMEM_CKPT" "$STMEM_SHA256"
python ecg-fms/_compute_stmem_embeddings.py \
--dataset-path $DATASET_PATH \
--output-path $RUN_DIR/$EXP_NAME \
--stmem-repo $REPO_ROOT/external/vuno-ST-MEM \
--stmem-ckpt $STMEM_CKPT

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
