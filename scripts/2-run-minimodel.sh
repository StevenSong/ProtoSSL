#!/bin/bash

set -e

ECHONEXT_DATA=/opt/gpudata/ecg/echonext
RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/runs
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-transfer
IMAGE_TAG=echonext-minimodel
GPU_IDX=1

# docker will create directories under root so premake any
# directories we need to write into outside of docker container
mkdir $RUN_DIR/echonext-minimodel

cd "$REPO_ROOT/external/PierreElias-IntroECG/7-EchoNext Minimodel"

docker build -t $IMAGE_TAG .

docker run --rm --gpus device=$GPU_IDX \
-v $ECHONEXT_DATA:/processed_data \
-v $RUN_DIR/echonext-minimodel:/results \
$IMAGE_TAG

cd "$REPO_ROOT/scripts"

python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/echonext-minimodel/prediction_loop/probs.npy \
--output-path $RUN_DIR/echonext-minimodel

ln -s ./prediction_loop/probs.npy \
$RUN_DIR/echonext-minimodel/probs.npy 
