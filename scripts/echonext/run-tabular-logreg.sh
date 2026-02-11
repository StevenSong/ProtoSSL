#!/bin/bash

set -e

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
echo "Using REPO_ROOT=$REPO_ROOT"
cd $REPO_ROOT/scripts

EXP_NAME=tabular-logreg

python echonext/_tabular_logreg.py \
--echonext-data $DATASET_PATH \
--output-path $RUN_DIR/$EXP_NAME-unweighted
python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME-unweighted/probs.npy \
--output-path $RUN_DIR/$EXP_NAME-unweighted

python echonext/_tabular_logreg.py \
--echonext-data $DATASET_PATH \
--balance-class-weight \
--output-path $RUN_DIR/$EXP_NAME-weighted
python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME-weighted/probs.npy \
--output-path $RUN_DIR/$EXP_NAME-weighted
