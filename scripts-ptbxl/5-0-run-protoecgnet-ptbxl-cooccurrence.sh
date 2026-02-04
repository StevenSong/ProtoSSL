#!/bin/bash

set -e

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
export SCP_GROUP_PATH="/opt/gpudata/steven/ecg-prototype-fm/external/bbj-lab-protoecgnet/scp_statementsRegrouped2.csv"
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm
cd $REPO_ROOT/scripts-ptbxl
PROTOECGNET_REPO=/opt/gpudata/steven/ecg-prototype-fm/external/bbj-lab-protoecgnet

cd $PROTOECGNET_REPO/src
python label_co.py --label-set all --save-path $RUN_DIR/protoecgnet-ptbxl-cooccurrence/label_cooccur_Cat1.pt
