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
cd $REPO_ROOT/scripts

# experiment parameters
EXP_NAME="pass-heedb-to-echonext-w-proj"
PRETRAIN_RUN="/opt/gpudata/steven/ecg-prototype-fm/outputs/runs/pass-pretrain-heedb"
N_PROTOTYPES=128

# this version relies on samples projected in the transfer dataset
# first project
python -m pass_pclr.trainer \
    --pipeline-stage project-prototypes \
    --config $REPO_ROOT/configs/pass-pclr.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $ECHONEXT_DATA \
    --model.n_prototypes $N_PROTOTYPES \
    --model.pretrained_weights $PRETRAIN_RUN/learn-prototypes/latest/best.ckpt

# then train classifier
python -m pass_pclr.trainer \
    --pipeline-stage train-classifier \
    --config $REPO_ROOT/configs/pass-pclr.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $ECHONEXT_DATA \
    --model.n_prototypes $N_PROTOTYPES \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes/latest/proj.ckpt

python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

ln -s ./train-classifier/latest/probs.npy \
$RUN_DIR/$EXP_NAME/probs.npy
