#!/bin/bash

set -e

ECHONEXT_DATA=/opt/gpudata/ecg/echonext
RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/runs-2 # running a parallel job with the same run dir seems to result in a low-level filesystem lock so this is just a quick, trivial hack to circumvent it
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-transfer
RESNET=resnet50
CONV=2D

python _echonext_resnet.py \
--config $REPO_ROOT/configs/resnet.yaml \
--trainer.logger.save_dir $RUN_DIR \
--trainer.logger.name $RESNET-$CONV \
--model.resnet_type $RESNET \
--model.conv_type $CONV

python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/$RESNET-$CONV/latest/probs.npy \
--output-path $RUN_DIR/$RESNET-$CONV

ln -s ./latest/probs.npy \
$RUN_DIR/$RESNET-$CONV/probs.npy
