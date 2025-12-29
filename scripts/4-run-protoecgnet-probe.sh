#!/bin/bash

set -e

ECHONEXT_DATA=/opt/gpudata/ecg/echonext
RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/runs
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-transfer
PROTOECGNET_REPO=/opt/gpudata/steven/ecg-prototype-transfer/external/bbj-lab-protoecgnet
PROTOECGNET_CKPT=/opt/gpudata/steven/ecg-prototype-transfer/external/bbj-lab-protoecgnet/ptbxl-classifier-checkpoints

################################################################################
# Embed Prototypes
################################################################################

python _protoecgnet_embed.py \
--echonext-data $ECHONEXT_DATA \
--protoecgnet-repo $PROTOECGNET_REPO \
--protoecgnet-checkpoints $PROTOECGNET_CKPT \
--output-path $RUN_DIR/protoecgnet-embeddings

################################################################################
# Probe Cat 1, No PCA, No Weighting
################################################################################

python _protoecgnet_probe.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type sim1d \
--output-path $RUN_DIR/proto-sim1d-logreg-unweighted
python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-sim1d-logreg-unweighted/probs.npy \
--output-path $RUN_DIR/proto-sim1d-logreg-unweighted

################################################################################
# Probe Cat 1, No PCA, With Weighting
################################################################################

python _protoecgnet_probe.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type sim1d \
--balance-class-weight \
--output-path $RUN_DIR/proto-sim1d-logreg-weighted
python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-sim1d-logreg-weighted/probs.npy \
--output-path $RUN_DIR/proto-sim1d-logreg-weighted

################################################################################
# Probe Cat 1, 32d PCA, No Weighting
################################################################################

python _protoecgnet_probe.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type sim1d \
--embedding-pca 32 \
--output-path $RUN_DIR/proto-sim1d-logreg-pca32
python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-sim1d-logreg-pca32/probs.npy \
--output-path $RUN_DIR/proto-sim1d-logreg-pca32

################################################################################
# Probe Cat 3, 64d PCA, No Weighting
################################################################################

python _protoecgnet_probe.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type sim2d_partial \
--embedding-pca 64 \
--output-path $RUN_DIR/proto-sim2d_partial-logreg-pca64
python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-sim2d_partial-logreg-pca64/probs.npy \
--output-path $RUN_DIR/proto-sim2d_partial-logreg-pca64

################################################################################
# Probe Cat 4, No PCA, No Weighting
################################################################################

python _protoecgnet_probe.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type sim2d_global \
--output-path $RUN_DIR/proto-sim2d_global-logreg
python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-sim2d_global-logreg/probs.npy \
--output-path $RUN_DIR/proto-sim2d_global-logreg

################################################################################
# Probe Fusion, 64d PCA, No Weighting
################################################################################

python _protoecgnet_probe.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--prototype-embeddings $RUN_DIR/protoecgnet-embeddings \
--embedding-type all \
--embedding-pca 64 \
--output-path $RUN_DIR/proto-all-logreg-pca64
python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/proto-all-logreg-pca64/probs.npy \
--output-path $RUN_DIR/proto-all-logreg-pca64


