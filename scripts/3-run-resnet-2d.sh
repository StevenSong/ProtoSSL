#!/bin/bash

set -e

# set these env vars prior to executing this script
# ECHONEXT_DATA=/opt/gpudata/ecg/echonext
# RUN_DIR=/opt/gpudata/steven/ecg-prototype-fm/runs-2 # running a parallel job with the same run dir seems to result in a low-level filesystem lock so this is just a quick, trivial hack to circumvent it
: "${ECHONEXT_DATA:?Env var ECHONEXT_DATA must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
echo "Using ECHONEXT_DATA=$ECHONEXT_DATA"
echo "Using RUN_DIR=$RUN_DIR-2 (note that this uses a suffixed path to avoid a filesystem deadlock)"
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm
cd $REPO_ROOT/scripts
RESNET=resnet50
CONV=2D

python _echonext_resnet.py \
--config $REPO_ROOT/configs/resnet.yaml \
--trainer.logger.save_dir $RUN_DIR-2 \
--trainer.logger.name $RESNET-$CONV \
--model.resnet_type $RESNET \
--model.conv_type $CONV \
--data.echonext_data $ECHONEXT_DATA

python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR-2/$RESNET-$CONV/latest/probs.npy \
--output-path $RUN_DIR-2/$RESNET-$CONV

ln -s ./latest/probs.npy \
$RUN_DIR-2/$RESNET-$CONV/probs.npy

mv $RUN_DIR-2/$RESNET-$CONV $RUN_DIR/$RESNET-$CONV
