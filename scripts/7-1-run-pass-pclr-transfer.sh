#!/bin/bash

set -e

# set these env vars prior to executing this script
# ECHONEXT_DATA=/opt/gpudata/ecg/echonext
# RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/outputs/runs
: "${ECHONEXT_DATA:?Env var ECHONEXT_DATA must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
echo "Using ECHONEXT_DATA=$ECHONEXT_DATA"
echo "Using RUN_DIR=$RUN_DIR"
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-transfer
cd $REPO_ROOT/scripts

# experiment parameters
EXP_NAME="pass-ptbxl-to-echonext"
PRETRAIN_RUN="/opt/gpudata/steven/ecg-prototype-transfer/outputs/runs/pass-pretrain-ptbxl"
N_PROTOTYPES=128

# this version relies on samples projected in the pretraining dataset
python -m pass_pclr.trainer main.py \
    --pipeline-stage train-classifier \
    --config $REPO_ROOT/configs/pass-pclr.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $ECHONEXT_DATA \
    --model.n_prototypes $N_PROTOTYPES \
    --model.pretrained_weights $PRETRAIN_RUN/project-prototypes/latest/proj.ckpt

python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME
