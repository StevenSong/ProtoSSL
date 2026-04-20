#!/bin/bash

set -e

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"
: "${SEED:=42}"
echo "Using SEED=$SEED"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
echo "Using REPO_ROOT=$REPO_ROOT"
cd $REPO_ROOT/scripts

# experiment parameters
EXP_NAME="labsup-proto-direct"

python -m protossl.trainer \
    --seed_everything $SEED \
    --pipeline-stage learn-prototypes-supervised \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --trainer.logger.save_dir $RUN_DIR/ \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH

python -m protossl.trainer \
    --seed_everything $SEED \
    --pipeline-stage project-prototypes-supervised \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --trainer.logger.save_dir $RUN_DIR/ \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototypes-supervised/latest/best.ckpt

python -m protossl.trainer \
    --seed_everything $SEED \
    --pipeline-stage compute-embeddings \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes-supervised/latest/proj.ckpt

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
