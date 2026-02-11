#!/bin/bash

set -e

# set these env vars prior to executing this script
# ECHONEXT_DATA=/opt/gpudata/ecg/echonext
# RUN_DIR=/opt/gpudata/steven/ecg-prototype-fm/outputs/runs
: "${ECHONEXT_DATA:?Env var ECHONEXT_DATA must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
echo "Using ECHONEXT_DATA=$ECHONEXT_DATA"
echo "Using RUN_DIR=$RUN_DIR"
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm
cd $REPO_ROOT/scripts/echonext

# experiment parameters
EXP_NAME="resnet50-1D"

# this version relies on samples projected in the pretraining dataset
python -m pass_pclr.trainer \
    --config $REPO_ROOT/configs/resnet.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $ECHONEXT_DATA

python _eval_echonext_probs.py \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

ln -s ./train-classifier/latest/probs.npy \
$RUN_DIR/$EXP_NAME/probs.npy
