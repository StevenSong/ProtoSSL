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
    --job_name tune_cat1 \
    --epochs 200 \
    --n_trials 200 \
    --checkpoint_dir $RUN_DIR/protoecgnet-echonext-cat1/checkpoints \
    --log_dir $RUN_DIR/protoecgnet-echonext-cat1/logs \
    --test_dir $RUN_DIR/protoecgnet-echonext-cat1/test_results \
    --study_dir $RUN_DIR/protoecgnet-echonext-cat1/optuna_studies \
    --sampling_rate 100 \
    --label_set "1" \
    --num_workers 4 \
    --dimension "1D" \
    --seed 42 \
    --training_stage "joint" \
    --custom_groups True \
    --proto_dim 2048 \
    --backbone resnet1d50 \
    --standardize False \
    --remove_baseline False

echo
echo "========================================================================="
echo "MUST NOW DO PROJECTION OF PROTOTYPES TO GROUND IN REAL DATA SAMPLES"
echo "========================================================================="
echo