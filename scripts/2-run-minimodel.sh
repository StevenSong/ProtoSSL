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
cd $REPO_ROOT/scripts
IMAGE_TAG=echonext-minimodel
: "${CUDA_VISIBLE_DEVICES:?Env var CUDA_VISIBLE_DEVICES must be set prior to script execution}"
GPU_IDX=$CUDA_VISIBLE_DEVICES

# docker will create directories under root so premake any
# directories we need to write into outside of docker container
mkdir -p $RUN_DIR/echonext-minimodel

cd "$REPO_ROOT/external/PierreElias-IntroECG/7-EchoNext Minimodel"

docker build -t $IMAGE_TAG .

docker run --rm --gpus device=$GPU_IDX \
-v $ECHONEXT_DATA:/processed_data \
-v $RUN_DIR/echonext-minimodel:/results \
$IMAGE_TAG

cd "$REPO_ROOT/scripts"

python _eval_echonext_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/echonext-minimodel/prediction_loop/probs.npy \
--output-path $RUN_DIR/echonext-minimodel

ln -s ./prediction_loop/probs.npy \
$RUN_DIR/echonext-minimodel/probs.npy 
