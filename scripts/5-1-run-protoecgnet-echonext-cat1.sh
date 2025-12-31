#!/bin/bash

set -e

ECHONEXT_DATA=/opt/gpudata/ecg/echonext
RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/runs
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-transfer
PROTOECGNET_REPO=/opt/gpudata/steven/ecg-prototype-transfer/external/bbj-lab-protoecgnet
N_TRIALS=100
BATCH_SIZE=2048
NUM_WORKERS=4

# Experiment parameters
ARCH=resnet1d18
PROTO_DIM=512
LABEL_SET=1
CONV_DIM=1D
EXP_DIR=$RUN_DIR/protoecgnet-echonext-cat1

echo
echo "========================================================================="
echo "ENSURE THAT proto_models1D.py AND proto_models2D.py IN $PROTOECGNET_REPO/src ARE MANUALLY SET TO USE COMPUTED CO-OCCURENCE MATRICES"
echo "========================================================================="
echo

cd $PROTOECGNET_REPO/src
python tune.py \
    --training_stage "joint" \
    --standardize False \
    --remove_baseline False \
    --job_name joint_cat"$LABEL_SET" \
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
    --custom_groups True \
    --proto_dim $PROTO_DIM \
    --backbone $ARCH \

echo
echo "========================================================================="
echo "PROJECTING PROTOTYPES TO GROUND IN REAL DATA SAMPLES"
echo "========================================================================="
echo

cd $REPO_ROOT/scripts
python _protoecgnet_postprocess_results.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--output-path $EXP_DIR/checkpoints/joint_cat"$LABEL_SET" \
--study-pkl $EXP_DIR/optuna_studies/joint_cat"$LABEL_SET"_optuna_study.pkl \
--trial-checkpoints $EXP_DIR/checkpoints/joint_cat"$LABEL_SET"

cd $PROTOECGNET_REPO/src
python3 main.py \
    --training_stage projection \
    --custom_groups True \
    --sampling_rate 100 \
    --standardize False \
    --remove_baseline False \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --proto_dim $PROTO_DIM \
    --seed 42 \
    --job_name proj_cat"$LABEL_SET" \
    --checkpoint_dir $EXP_DIR/checkpoints \
    --log_dir $EXP_DIR/logs \
    --pretrained_weights $EXP_DIR/checkpoints/joint_cat"$LABEL_SET"/best.ckpt \
    --dimension $CONV_DIM \
    --backbone $ARCH \
    --label_set $LABEL_SET

echo
echo "========================================================================="
echo "TUNING CLASSIFIER USING FROZEN FEATURE EXTRACTOR AND PROTOTYPES"
echo "========================================================================="
echo

cd $PROTOECGNET_REPO/src
python tune.py \
    --training_stage "classifier" \
    --job_name cls_cat"$LABEL_SET" \
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
    --custom_groups True \
    --proto_dim $PROTO_DIM \
    --backbone $ARCH \
    --standardize False \
    --remove_baseline False \
    --pretrained_weights $EXP_DIR/checkpoints/proj_cat"$LABEL_SET"/proj_cat"$LABEL_SET"_projection.pth

cd $REPO_ROOT/scripts
python _protoecgnet_postprocess_results.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--output-path $EXP_DIR \
--study-pkl $EXP_DIR/optuna_studies/cls_cat"$LABEL_SET"_optuna_study.pkl \
--trial-predictions $EXP_DIR/test_results/cls_cat"$LABEL_SET"
python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $EXP_DIR/probs.npy \
--output-path $EXP_DIR
