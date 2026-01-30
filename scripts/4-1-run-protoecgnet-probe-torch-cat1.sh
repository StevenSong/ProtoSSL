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
PROTOECGNET_REPO=/opt/gpudata/steven/ecg-prototype-fm/external/bbj-lab-protoecgnet
N_TRIALS=100
BATCH_SIZE=2048
NUM_WORKERS=4

# Experiment parameters
ARCH=resnet1d18
PROTO_DIM=512
LABEL_SET=1
CONV_DIM=1D
EXP_DIR=$RUN_DIR/proto-ptbxl-pip-cat1

echo
echo "========================================================================="
echo "ENSURE THAT \`DATASET_PATH\` and \`SCP_GROUP_PATH\` IN $PROTOECGNET_REPO/src/ecg_utils.py ARE MANUALLY SET BEFORE EXECUTING THIS SCRIPT"
echo "========================================================================="
echo
echo
echo "========================================================================="
echo "CAN SAFELY IGNORE WARNING ABOUT MISSING COOCCURRENCE MATRIX DURING CLASSIFIER TRAINING"
echo "========================================================================="
echo

################################################################################
# Tune Cat 1
################################################################################

cd $PROTOECGNET_REPO/src
python tune.py \
    --job_name probe_cat$LABEL_SET \
    --epochs 200 \
    --batch_size $BATCH_SIZE \
    --n_trials $N_TRIALS \
    --checkpoint_dir $EXP_DIR/checkpoints \
    --log_dir $EXP_DIR/logs \
    --test_dir $EXP_DIR/test_results \
    --study_dir $EXP_DIR/optuna_studies \
    --sampling_rate 100 \
    --label_set $LABEL_SET \
    --num_workers $NUM_WORKERS \
    --dimension $CONV_DIM \
    --seed 42 \
    --training_stage "classifier" \
    --custom_groups True \
    --proto_dim $PROTO_DIM \
    --backbone $ARCH \
    --standardize False \
    --remove_baseline False \
    --pretrained_weights $PROTOECGNET_REPO/ptbxl-classifier-checkpoints/cat$LABEL_SET.ckpt

cd $REPO_ROOT/scripts
python _protoecgnet_postprocess_echonext_results.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--output-path $EXP_DIR \
--study-pkl $EXP_DIR/optuna_studies/probe_cat"$LABEL_SET"_optuna_study.pkl \
--trial-predictions $EXP_DIR/test_results/probe_cat$LABEL_SET
python _eval_echonext_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $EXP_DIR/probs.npy \
--output-path $EXP_DIR
