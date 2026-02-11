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
PROTOECGNET_REPO=/opt/gpudata/steven/ecg-prototype-fm/external/bbj-lab-protoecgnet
PROTOECGNET_CKPT=/opt/gpudata/steven/ecg-prototype-fm/external/bbj-lab-protoecgnet/ptbxl-classifier-checkpoints

################################################################################
# Embed Prototypes
################################################################################

echo
echo "========================================================================="
echo "CAN SAFELY IGNORE WARNING ABOUT MISSING COOCCURRENCE MATRIX FOR INFERENCE"
echo "========================================================================="
echo

python _protoecgnet_embed_echonext.py \
--echonext-data $ECHONEXT_DATA \
--protoecgnet-repo $PROTOECGNET_REPO \
--protoecgnet-checkpoints $PROTOECGNET_CKPT \
--output-path $RUN_DIR/protoecgnet-embeddings

################################################################################
# Probe Cat 1, No PCA, No Weighting
################################################################################

python _protoecgnet_linear_probe_echonext.py \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type sim1d \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-cat1-unweighted
python _eval_echonext_probs.py \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-ptbxl-pip-logreg-cat1-unweighted/probs.npy \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-cat1-unweighted

################################################################################
# Probe Cat 1, No PCA, With Weighting
################################################################################

python _protoecgnet_linear_probe_echonext.py \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type sim1d \
--balance-class-weight \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-cat1-weighted
python _eval_echonext_probs.py \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-ptbxl-pip-logreg-cat1-weighted/probs.npy \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-cat1-weighted

################################################################################
# Probe Cat 1, 32d PCA, No Weighting
################################################################################

python _protoecgnet_linear_probe_echonext.py \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type sim1d \
--embedding-pca 32 \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-cat1-pca32
python _eval_echonext_probs.py \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-ptbxl-pip-logreg-cat1-pca32/probs.npy \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-cat1-pca32

################################################################################
# Probe Cat 3, 64d PCA, No Weighting
################################################################################

python _protoecgnet_linear_probe_echonext.py \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type sim2d_partial \
--embedding-pca 64 \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-cat3-pca64
python _eval_echonext_probs.py \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-ptbxl-pip-logreg-cat3-pca64/probs.npy \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-cat3-pca64

################################################################################
# Probe Cat 4, No PCA, No Weighting
################################################################################

python _protoecgnet_linear_probe_echonext.py \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type sim2d_global \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-cat4
python _eval_echonext_probs.py \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-ptbxl-pip-logreg-cat4/probs.npy \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-cat4

################################################################################
# Probe Fusion, 64d PCA, No Weighting
################################################################################

python _protoecgnet_linear_probe_echonext.py \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type all \
--embedding-pca 64 \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-fusion-pca64
python _eval_echonext_probs.py \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-ptbxl-pip-logreg-fusion-pca64/probs.npy \
--output-path $RUN_DIR/proto-ptbxl-pip-logreg-fusion-pca64
