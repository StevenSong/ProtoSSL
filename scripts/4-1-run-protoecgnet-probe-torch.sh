#!/bin/bash

set -e

ECHONEXT_DATA=/opt/gpudata/ecg/echonext
RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/runs
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-transfer
PROTOECGNET_REPO=/opt/gpudata/steven/ecg-prototype-transfer/external/bbj-lab-protoecgnet
N_TRIALS=200
BATCH_SIZE=2048

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
    --job_name probe_cat1 \
    --epochs 200 \
    --batch_size $BATCH_SIZE \
    --n_trials $N_TRIALS \
    --checkpoint_dir $RUN_DIR/protoecgnet-transfer-cat1/checkpoints \
    --log_dir $RUN_DIR/protoecgnet-transfer-cat1/logs \
    --test_dir $RUN_DIR/protoecgnet-transfer-cat1/test_results \
    --study_dir $RUN_DIR/protoecgnet-transfer-cat1/optuna_studies \
    --sampling_rate 100 \
    --label_set "1" \
    --num_workers 4 \
    --dimension "1D" \
    --seed 42 \
    --training_stage "classifier" \
    --custom_groups True \
    --proto_dim 512 \
    --backbone resnet1d18 \
    --standardize False \
    --remove_baseline False \
    --pretrained_weights $PROTOECGNET_REPO/ptbxl-classifier-checkpoints/cat1.ckpt

cd $REPO_ROOT/scripts
python _protoecgnet_postprocess_results.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--output-path $RUN_DIR/protoecgnet-transfer-cat1 \
--study-pkl $RUN_DIR/protoecgnet-transfer-cat1/optuna_studies/probe_cat1_optuna_study.pkl \
--trial-predictions $RUN_DIR/protoecgnet-transfer-cat1/test_results/probe_cat1
python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/protoecgnet-transfer-cat1/probs.npy \
--output-path $RUN_DIR/protoecgnet-transfer-cat1

################################################################################
# Tune Cat 3
################################################################################

cd $PROTOECGNET_REPO/src
python tune.py \
    --job_name probe_cat3 \
    --epochs 200 \
    --batch_size $BATCH_SIZE \
    --n_trials $N_TRIALS \
    --checkpoint_dir $RUN_DIR/protoecgnet-transfer-cat3/checkpoints \
    --log_dir $RUN_DIR/protoecgnet-transfer-cat3/logs \
    --test_dir $RUN_DIR/protoecgnet-transfer-cat3/test_results \
    --study_dir $RUN_DIR/protoecgnet-transfer-cat3/optuna_studies \
    --sampling_rate 100 \
    --label_set "3" \
    --num_workers 4 \
    --dimension "2D" \
    --seed 42 \
    --training_stage "classifier" \
    --custom_groups True \
    --proto_dim 512 \
    --proto_time_len 3 \
    --backbone resnet18 \
    --standardize False \
    --remove_baseline False \
    --pretrained_weights $PROTOECGNET_REPO/ptbxl-classifier-checkpoints/cat3.ckpt

cd $REPO_ROOT/scripts
python _protoecgnet_postprocess_results.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--output-path $RUN_DIR/protoecgnet-transfer-cat3 \
--study-pkl $RUN_DIR/protoecgnet-transfer-cat3/optuna_studies/probe_cat3_optuna_study.pkl \
--trial-predictions $RUN_DIR/protoecgnet-transfer-cat3/test_results/probe_cat3
python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/protoecgnet-transfer-cat3/probs.npy \
--output-path $RUN_DIR/protoecgnet-transfer-cat3

################################################################################
# Tune Cat 4
################################################################################

cd $PROTOECGNET_REPO/src
python tune.py \
    --job_name probe_cat4 \
    --epochs 200 \
    --batch_size $BATCH_SIZE \
    --n_trials $N_TRIALS \
    --checkpoint_dir $RUN_DIR/protoecgnet-transfer-cat4/checkpoints \
    --log_dir $RUN_DIR/protoecgnet-transfer-cat4/logs \
    --test_dir $RUN_DIR/protoecgnet-transfer-cat4/test_results \
    --study_dir $RUN_DIR/protoecgnet-transfer-cat4/optuna_studies \
    --sampling_rate 100 \
    --label_set "4" \
    --num_workers 4 \
    --dimension "2D" \
    --seed 42 \
    --training_stage "classifier" \
    --custom_groups True \
    --proto_dim 512 \
    --proto_time_len 32 \
    --backbone resnet18 \
    --standardize False \
    --remove_baseline False \
    --pretrained_weights $PROTOECGNET_REPO/ptbxl-classifier-checkpoints/cat4.ckpt

cd $REPO_ROOT/scripts
python _protoecgnet_postprocess_results.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--output-path $RUN_DIR/protoecgnet-transfer-cat4 \
--study-pkl $RUN_DIR/protoecgnet-transfer-cat4/optuna_studies/probe_cat4_optuna_study.pkl \
--trial-predictions $RUN_DIR/protoecgnet-transfer-cat4/test_results/probe_cat4
python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/protoecgnet-transfer-cat4/probs.npy \
--output-path $RUN_DIR/protoecgnet-transfer-cat4

################################################################################
# Tune Fusion
################################################################################

cd $PROTOECGNET_REPO/src
python tune.py \
    --job_name probe_fusion \
    --epochs 200 \
    --batch_size $BATCH_SIZE \
    --n_trials $N_TRIALS \
    --checkpoint_dir $RUN_DIR/protoecgnet-transfer-fusion/checkpoints \
    --log_dir $RUN_DIR/protoecgnet-transfer-fusion/logs \
    --test_dir $RUN_DIR/protoecgnet-transfer-fusion/test_results \
    --study_dir $RUN_DIR/protoecgnet-transfer-fusion/optuna_studies \
    --sampling_rate 100 \
    --label_set "4" \
    --num_workers 4 \
    --dimension "2D" \
    --seed 42 \
    --training_stage "fusion" \
    --custom_groups True \
    --proto_dim 512 \
    --proto_time_len 3 \
    --backbone resnet18 \
    --standardize False \
    --remove_baseline False \
    --fusion_weights1 $PROTOECGNET_REPO/ptbxl-classifier-checkpoints/cat1.ckpt \
    --fusion_weights3 $PROTOECGNET_REPO/ptbxl-classifier-checkpoints/cat3.ckpt \
    --fusion_weights4 $PROTOECGNET_REPO/ptbxl-classifier-checkpoints/cat4.ckpt \
    --fusion_backbone1 resnet1d18 \
    --fusion_backbone3 resnet18 \
    --fusion_backbone4 resnet18 \
    --fusion_proto_dim1 512 \
    --fusion_proto_dim3 512 \
    --fusion_proto_dim4 512 \
    --fusion_single_ppc1 5 \
    --fusion_single_ppc3 18 \
    --fusion_single_ppc4 7 \
    --fusion_joint_ppb1 0 \
    --fusion_joint_ppb3 0 \
    --fusion_joint_ppb4 0

cd $REPO_ROOT/scripts
python _protoecgnet_postprocess_results.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--output-path $RUN_DIR/protoecgnet-transfer-fusion \
--study-pkl $RUN_DIR/protoecgnet-transfer-fusion/optuna_studies/probe_fusion_optuna_study.pkl \
--trial-predictions $RUN_DIR/protoecgnet-transfer-fusion/test_results/probe_fusion
python _eval_probs.py \
--target-config $REPO_ROOT/configs/targets.yaml \
--echonext-data $ECHONEXT_DATA \
--probs-npy $RUN_DIR/protoecgnet-transfer-fusion/probs.npy \
--output-path $RUN_DIR/protoecgnet-transfer-fusion
