#!/usr/bin/env bash
# Columbia's EchoNext minimodel (external/PierreElias-IntroECG), in its own Docker image.
#   slurm/submit.sh -p h200 --gres=gpu:1 -- scripts/echonext/run-columbia-minimodel.sh <dataset> <run> <seed>
#   <dataset>: echonext or a subset, e.g. echonext-256 (<seed> only names the output dir)
set -euo pipefail
source "$(dirname "$0")/../_common.sh" "$@"

EXP_NAME=columbia-minimodel
IMAGE_TAG=echonext-minimodel

stage "$EXP_NAME"

# docker will create directories under root so premake any
# directories we need to write into outside of docker container
mkdir -p $RUN_DIR/$EXP_NAME

cd "$REPO_ROOT/external/PierreElias-IntroECG/7-EchoNext Minimodel"
# one build per node at a time: concurrent builds of a tag each retag it, so a job could
# run the other's image
exec 8>"$RUN_ROOT/docker-build.lock"
flock 8
docker build -t $IMAGE_TAG .
image=$(docker image inspect --format '{{.Id}}' $IMAGE_TAG)
flock -u 8
echo "image $image"
docker run --rm --gpus device=$CUDA_VISIBLE_DEVICES \
-v $DATASET_PATH:/processed_data \
-v $RUN_DIR/$EXP_NAME:/results \
$image

cd "$REPO_ROOT/scripts"
python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/prediction_loop/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

cp $RUN_DIR/$EXP_NAME/prediction_loop/probs.npy $RUN_DIR/$EXP_NAME/probs.npy

publish "$EXP_NAME"
