#!/usr/bin/env bash
#   slurm/submit.sh --gres=gpu:1 -- scripts/9-2-run-ecgfounder-clustering.sh <dataset> <run> <seed>
# see scripts/_common.sh for the arguments and where outputs go
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

# align with other experiments
PPL=14

# experiment parameters
EXP_NAME="ecgfounder-clustering"

stage "$EXP_NAME"

# node-local, shared by 9-1..9-3 on this node
bash 9-0-run-ecgfounder-patches.sh "$@"

python prototypes-from-fms/SK_OT_prototype_clustering.py \
--random-seed $SEED \
--dataset-path $DATASET_PATH \
--patch-embeddings $RUN_DIR/ecgfounder-patches \
--prototypes-per-label $PPL \
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
