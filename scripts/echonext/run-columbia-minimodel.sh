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

# run using docker
IMAGE_TAG=echonext-minimodel
: "${CUDA_VISIBLE_DEVICES:?Env var CUDA_VISIBLE_DEVICES must be set prior to script execution}"
GPU_IDX=$CUDA_VISIBLE_DEVICES

# docker will create directories under root so premake any
# directories we need to write into outside of docker container
mkdir -p $RUN_DIR/columbia-minimodel

cd "$REPO_ROOT/external/PierreElias-IntroECG/7-EchoNext Minimodel"
docker build -t $IMAGE_TAG .
docker run --rm --gpus device=$GPU_IDX \
-v $DATASET_PATH:/processed_data \
-v $RUN_DIR/columbia-minimodel:/results \
$IMAGE_TAG

cd "$REPO_ROOT/scripts"
python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/columbia-minimodel/prediction_loop/probs.npy \
--output-path $RUN_DIR/columbia-minimodel

cp $RUN_DIR/columbia-minimodel/prediction_loop/probs.npy $RUN_DIR/columbia-minimodel/probs.npy 
