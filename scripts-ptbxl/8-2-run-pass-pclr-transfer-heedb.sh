#!/bin/bash

set -e

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm
cd $REPO_ROOT/scripts-ptbxl

# experiment parameters
EXP_NAME="pass-heedb-pip"
PRETRAIN_RUN="/opt/gpudata/steven/ecg-prototype-fm/outputs/runs/pass-pretrain-heedb"

# this version relies on samples projected in the pretraining dataset
python -m pass_pclr.trainer \
    --pipeline-stage train-classifier \
    --config $REPO_ROOT/configs/pass-pclr.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $PRETRAIN_RUN/project-prototypes/latest/proj.ckpt

python _eval_ptbxl_probs.py \
--ptbxl-data $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

ln -s ./train-classifier/latest/probs.npy \
$RUN_DIR/$EXP_NAME/probs.npy
