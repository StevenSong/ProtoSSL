#!/bin/bash

set -e

ECHONEXT_DATA=/opt/gpudata/ecg/echonext
RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/runs
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-transfer
PROTOECGNET_REPO=/opt/gpudata/steven/ecg-prototype-transfer/external/bbj-lab-protoecgnet

echo
echo "========================================================================="
echo "ENSURE THAT proto_models1D.py AND proto_models2D.py IN $PROTOECGNET_REPO/src ARE MANUALLY SET TO USE COMPUTED CO-OCCURENCE MATRICES"
echo "========================================================================="
echo

cd $PROTOECGNET_REPO/src
python tune.py \
    --job_name tune_cat4 \
    --epochs 200 \
    --n_trials 200 \
    --checkpoint_dir $RUN_DIR/protoecgnet-echonext-cat4/checkpoints \
    --log_dir $RUN_DIR/protoecgnet-echonext-cat4/logs \
    --test_dir $RUN_DIR/protoecgnet-echonext-cat4/test_results \
    --study_dir $RUN_DIR/protoecgnet-echonext-cat4/optuna_studies \
    --sampling_rate 100 \
    --label_set "4" \
    --num_workers 4 \
    --dimension "2D" \
    --seed 42 \
    --training_stage "joint" \
    --custom_groups True \
    --proto_dim 2048 \
    --proto_time_len 32 \
    --backbone resnet50 \
    --standardize False \
    --remove_baseline False

echo
echo "========================================================================="
echo "MUST NOW DO PROJECTION OF PROTOTYPES TO GROUND IN REAL DATA SAMPLES"
echo "========================================================================="
echo