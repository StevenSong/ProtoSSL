#!/bin/bash

set -e

# set these env vars prior to executing this script
# ECHONEXT_DATA=/opt/gpudata/ecg/echonext
# RUN_DIR=/opt/gpudata/steven/ecg-prototype-fm/runs
: "${ECHONEXT_DATA:?Env var ECHONEXT_DATA must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
echo "Using ECHONEXT_DATA=$ECHONEXT_DATA"
echo "Using RUN_DIR=$RUN_DIR"
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm
cd $REPO_ROOT/scripts/echonext

EXP_NAME=tabular-logreg

python _logreg_echonext.py \
--echonext-data $ECHONEXT_DATA \
--output-path $RUN_DIR/$EXP_NAME-unweighted
python _eval_echonext_probs.py \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/$EXP_NAME-unweighted/probs.npy \
--output-path $RUN_DIR/$EXP_NAME-unweighted

python _logreg_echonext.py \
--echonext-data $ECHONEXT_DATA \
--balance-class-weight \
--output-path $RUN_DIR/$EXP_NAME-weighted
python _eval_echonext_probs.py \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/$EXP_NAME-weighted/probs.npy \
--output-path $RUN_DIR/$EXP_NAME-weighted
