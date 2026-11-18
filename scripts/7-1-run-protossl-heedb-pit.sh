#!/usr/bin/env bash
#   slurm/submit.sh --gres=gpu:1 -- scripts/7-1-run-protossl-heedb-pit.sh <dataset> <run> <seed>
# see scripts/_common.sh for the arguments and where outputs go
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

# experiment parameters
EXP_NAME="protossl-heedb-pit"
PRETRAIN_RUN=$PRETRAIN_DIR/protossl-heedb

stage "$EXP_NAME"

python -m protossl.trainer \
    --config $REPO_ROOT/configs/target-unguided.yaml \
    --seed_everything $SEED \
    --pipeline-stage project-prototypes \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $PRETRAIN_RUN/learn-prototypes/latest/best.ckpt

python -m protossl.trainer \
    --config $REPO_ROOT/configs/target-unguided.yaml \
    --seed_everything $SEED \
    --pipeline-stage compute-embeddings \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes/latest/proj.ckpt

python _linear_probe.py \
--random-seed $SEED \
--dataset-path $DATASET_PATH \
--prototype-embeddings $RUN_DIR/$EXP_NAME/compute-embeddings/latest \
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
