#!/bin/bash

set -e

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm
cd $REPO_ROOT/scripts/ptbxl

# experiment parameters
EXP_NAME="pass-heedb-pit-logreg"
PRETRAIN_RUN="$RUN_DIR/pass-heedb-pit"

# this version relies on samples projected in the transfer dataset
python -m pass_pclr.trainer \
    --pipeline-stage compute-embeddings \
    --config $REPO_ROOT/configs/pass-pclr.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $PRETRAIN_RUN/project-prototypes/latest/proj.ckpt

python _pass_pclr_linear_probe_ptbxl.py \
--ptbxl-data $DATASET_PATH \
--prototype-embeddings $RUN_DIR/$EXP_NAME/compute-embeddings/latest \
--output-path $RUN_DIR/$EXP_NAME

python _eval_ptbxl_probs.py \
--ptbxl-data $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
--output-path $RUN_DIR/$EXP_NAME
