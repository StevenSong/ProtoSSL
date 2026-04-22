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

export DATASET_PATH=/opt/gpudata/audioset
export RUN_DIR=/opt/gpu_working/steven/protossl-audio
export REPO_ROOT=/opt/gpu_working/steven/ProtoSSL

: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"

echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
echo "Using REPO_ROOT=$REPO_ROOT"

cd $REPO_ROOT/scripts

# experiment parameters
EXP_NAME="prosup-audioset"

echo "======================================"
echo "Stage 1: learn-prototypes-supervised"
echo "======================================"

srun python -m pass_pclr.trainer \
    --pipeline-stage learn-prototypes-supervised \
    --config $REPO_ROOT/configs/audio/pretrain-supervised.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH

echo "======================================"
echo "Stage 2: project prototypes supervised"
echo "======================================"

srun python -m pass_pclr.trainer \
    --pipeline-stage project-prototypes-supervised \
    --config $REPO_ROOT/configs/audio/pretrain-supervised.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototypes-supervised/latest/best.ckpt
