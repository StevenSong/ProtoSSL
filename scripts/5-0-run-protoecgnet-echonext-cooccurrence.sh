#!/bin/bash

set -e

# set these env vars prior to executing this script
# ECHONEXT_DATA=/opt/gpudata/ecg/echonext
# RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/runs
: "${ECHONEXT_DATA:?Env var ECHONEXT_DATA must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
echo "Using ECHONEXT_DATA=$ECHONEXT_DATA"
echo "Using RUN_DIR=$RUN_DIR"
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-transfer
cd $REPO_ROOT/scripts
PROTOECGNET_REPO=/opt/gpudata/steven/ecg-prototype-transfer/external/bbj-lab-protoecgnet

echo
echo "========================================================================="
echo "ENSURE THAT \`DATASET_PATH\` and \`SCP_GROUP_PATH\` IN $PROTOECGNET_REPO/src/ecg_utils.py ARE MANUALLY SET BEFORE EXECUTING THIS SCRIPT"
echo "========================================================================="
echo

cd $PROTOECGNET_REPO/src
python label_co.py --label-set 1 --save-path $RUN_DIR/protoecgnet-echonext-cooccurrence/label_cooccur_Cat1.pt
python label_co.py --label-set 3 --save-path $RUN_DIR/protoecgnet-echonext-cooccurrence/label_cooccur_Cat3.pt
python label_co.py --label-set 4 --save-path $RUN_DIR/protoecgnet-echonext-cooccurrence/label_cooccur_Cat4.pt

echo
echo "========================================================================="
echo "MUST NOW MANUALLY UPDATE proto_models1D.py AND proto_models2D.py IN $PROTOECGNET_REPO/src TO USE CO-OCCURRENCE MATRICES SAVED IN $RUN_DIR/protoecgnet-echonext-cooccurrence"
echo "========================================================================="
echo
