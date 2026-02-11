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
cd $REPO_ROOT/scripts/echonext
PROTOECGNET_REPO=/opt/gpudata/steven/ecg-prototype-fm/external/bbj-lab-protoecgnet
N_TRIALS=100
BATCH_SIZE=2048
NUM_WORKERS=4

# Experiment parameters
CAT1_ARCH=resnet1d18
CAT1_PROTO_DIM=512

CAT3_ARCH=resnet18
CAT3_PROTO_DIM=512

CAT4_ARCH=resnet18
CAT4_PROTO_DIM=512

EXP_DIR=$RUN_DIR/proto-ptbxl-pit-fusion

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
# Tune Fusion
################################################################################

cd $PROTOECGNET_REPO/src
python tune.py \
    --job_name probe_fusion \
    --epochs 200 \
    --batch_size $BATCH_SIZE \
    --n_trials $N_TRIALS \
    --checkpoint_dir $EXP_DIR/checkpoints \
    --log_dir $EXP_DIR/logs \
    --test_dir $EXP_DIR/test_results \
    --study_dir $EXP_DIR/optuna_studies \
    --sampling_rate 100 \
    --num_workers $NUM_WORKERS \
    --seed 42 \
    --custom_groups True \
    --label_set 1 \
    --training_stage "fusion" \
    --standardize False \
    --remove_baseline False \
    --fusion_weights1 $RUN_DIR/protoecgnet-reproj-cat1/checkpoints/proj_cat1/proj_cat1_projection.pth \
    --fusion_weights3 $RUN_DIR/protoecgnet-reproj-cat3/checkpoints/proj_cat3/proj_cat3_projection.pth \
    --fusion_weights4 $RUN_DIR/protoecgnet-reproj-cat4/checkpoints/proj_cat4/proj_cat4_projection.pth \
    --fusion_backbone1 $CAT1_ARCH \
    --fusion_backbone3 $CAT3_ARCH \
    --fusion_backbone4 $CAT4_ARCH \
    --fusion_proto_dim1 $CAT1_PROTO_DIM \
    --fusion_proto_dim3 $CAT3_PROTO_DIM \
    --fusion_proto_dim4 $CAT4_PROTO_DIM \
    --proto_time_len 3 # this is for cat3 branch

cd $REPO_ROOT/scripts/echonext
python _protoecgnet_postprocess_echonext_results.py \
--output-path $EXP_DIR \
--study-pkl $EXP_DIR/optuna_studies/probe_fusion_optuna_study.pkl \
--trial-predictions $EXP_DIR/test_results/probe_fusion
python _eval_echonext_probs.py \
--echonext-data $ECHONEXT_DATA \
--probs-npy $EXP_DIR/probs.npy \
--output-path $EXP_DIR
