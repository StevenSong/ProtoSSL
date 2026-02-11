#!/bin/bash

set -e

PRETRAIN_DATASET=/opt/gpudata/ecg/heedb
RUN_DIR=/opt/gpudata/steven/ecg-prototype-fm/outputs
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm

# set these env vars prior to executing this script
: "${PRETRAIN_DATASET:?Env var PRETRAIN_DATASET must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"
echo "Using PRETRAIN_DATASET=$PRETRAIN_DATASET"
echo "Using RUN_DIR=$RUN_DIR"
echo "Using REPO_ROOT=$REPO_ROOT"
cd $REPO_ROOT/scripts

# experiment parameters
EXP_NAME="pass-pretrain-heedb"

# pretrain via self supervised prototype learning
python -m pass_pclr.trainer \
    --pipeline-stage learn-prototypes \
    --config $REPO_ROOT/configs/pass-pclr.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $PRETRAIN_DATASET \
    --data.num_workers 8 \
    --data.prefetch_factor 4

# project in the pretraining dataset
python -m pass_pclr.trainer \
    --pipeline-stage project-prototypes \
    --config $REPO_ROOT/configs/pass-pclr.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $PRETRAIN_DATASET \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototypes/latest/best.ckpt \
    --data.num_workers 8 \
    --data.prefetch_factor 4
