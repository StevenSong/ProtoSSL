#!/bin/bash

set -e

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
export SCP_GROUP_PATH="/opt/gpudata/steven/ecg-prototype-fm/external/bbj-lab-protoecgnet/scp_statementsRegrouped2.csv"
export COOC_DIR="protoecgnet-ptbxl-cooccurrence"
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm
cd $REPO_ROOT/scripts-ptbxl
PROTOECGNET_REPO=/opt/gpudata/steven/ecg-prototype-fm/external/bbj-lab-protoecgnet
N_TRIALS=100
BATCH_SIZE=2048
NUM_WORKERS=4

# Experiment parameters
ARCH=resnet18
PROTO_DIM=512
PROTO_TIME_LEN=3
LABEL_SET=all
CONV_DIM=2D
EXP_DIR=$RUN_DIR/proto-from-scratch-partial


cd $PROTOECGNET_REPO/src
python tune.py \
    --training_stage "joint" \
    --standardize True \
    --remove_baseline True \
    --job_name joint_"$LABEL_SET" \
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
    --proto_dim $PROTO_DIM \
    --proto_time_len $PROTO_TIME_LEN \
    --backbone $ARCH

echo
echo "========================================================================="
echo "PROJECTING PROTOTYPES TO GROUND IN REAL DATA SAMPLES"
echo "========================================================================="
echo

cd $REPO_ROOT/scripts-ptbxl
python _protoecgnet_postprocess_ptbxl_results.py \
--output-path $EXP_DIR/checkpoints/joint_"$LABEL_SET" \
--study-pkl $EXP_DIR/optuna_studies/joint_"$LABEL_SET"_optuna_study.pkl \
--trial-checkpoints $EXP_DIR/checkpoints/joint_"$LABEL_SET"

cd $PROTOECGNET_REPO/src
python3 main.py \
    --training_stage projection \
    --standardize True \
    --remove_baseline True \
    --job_name proj_"$LABEL_SET" \
    --batch_size $BATCH_SIZE \
    --checkpoint_dir $EXP_DIR/checkpoints \
    --log_dir $EXP_DIR/logs \
    --sampling_rate 100 \
    --label_set $LABEL_SET \
    --num_workers $NUM_WORKERS \
    --dimension $CONV_DIM \
    --seed 42 \
    --proto_dim $PROTO_DIM \
    --proto_time_len $PROTO_TIME_LEN \
    --backbone $ARCH \
    --pretrained_weights $EXP_DIR/checkpoints/joint_"$LABEL_SET"/best.ckpt

echo
echo "========================================================================="
echo "TUNING CLASSIFIER USING FROZEN FEATURE EXTRACTOR AND PROTOTYPES"
echo "========================================================================="
echo

cd $PROTOECGNET_REPO/src
python tune.py \
    --training_stage "classifier" \
    --standardize True \
    --remove_baseline True \
    --job_name cls_"$LABEL_SET" \
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
    --proto_dim $PROTO_DIM \
    --proto_time_len $PROTO_TIME_LEN \
    --backbone $ARCH \
    --pretrained_weights $EXP_DIR/checkpoints/proj_"$LABEL_SET"/proj_"$LABEL_SET"_projection.pth

cd $REPO_ROOT/scripts-ptbxl
python _protoecgnet_postprocess_ptbxl_results.py \
--output-path $EXP_DIR \
--study-pkl $EXP_DIR/optuna_studies/cls_"$LABEL_SET"_optuna_study.pkl \
--trial-predictions $EXP_DIR/test_results/cls_"$LABEL_SET"
python _eval_ptbxl_probs.py \
--ptbxl-data $DATASET_PATH \
--probs-npy $EXP_DIR/probs.npy \
--output-path $EXP_DIR
