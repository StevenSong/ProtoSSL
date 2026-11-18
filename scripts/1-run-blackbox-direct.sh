#!/usr/bin/env bash
#   slurm/submit.sh --gres=gpu:1 -- scripts/1-run-blackbox-direct.sh <dataset> <run> <seed>
# see scripts/_common.sh for the arguments and where outputs go
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

# experiment parameters
EXP_NAME="blackbox-direct"

stage "$EXP_NAME"

python -m protossl.trainer \
    --config $REPO_ROOT/configs/target-blackbox.yaml \
    --seed_everything $SEED \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH

cp $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy $RUN_DIR/$EXP_NAME/probs.npy

python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

python _eval_probs_bootstrapped.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

publish "$EXP_NAME"
