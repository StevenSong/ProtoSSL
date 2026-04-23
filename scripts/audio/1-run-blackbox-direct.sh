#!/bin/bash

#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=200gb
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH -w kg35-nvl01
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --output /home/songs1/slurm-logs/audio-%j.out

set -e

export REPO_ROOT=/opt/gpu_working/steven/ProtoSSL

# export ESC_TEST_FOLD=0
# export DATASET_PATH=/opt/gpudata/audio/ESC-50
# export RUN_DIR=/opt/gpu_working/steven/protossl-audio/runs-esc50-fold$ESC_TEST_FOLD

# export US8K_TEST_FOLD=0
# export DATASET_PATH=/opt/gpudata/audio/UrbanSound8K
# export RUN_DIR=/opt/gpu_working/steven/protossl-audio/runs-us8k-fold$US8K_TEST_FOLD

# export DATASET_PATH=/opt/gpudata/audio/speech-commands-v2
# export RUN_DIR=/opt/gpu_working/steven/protossl-audio/runs-speechcmds

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"
: "${SEED:=42}"
echo "Using SEED=$SEED"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
echo "Using REPO_ROOT=$REPO_ROOT"
cd $REPO_ROOT/scripts/audio

# experiment parameters
EXP_NAME="blackbox-direct"

python -m protossl.trainer \
    --seed_everything $SEED \
    --config $REPO_ROOT/configs/audio/target-blackbox.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH

cp $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy $RUN_DIR/$EXP_NAME/probs.npy

python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME
