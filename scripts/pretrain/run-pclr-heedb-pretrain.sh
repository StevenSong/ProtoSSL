#!/bin/bash

#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=500gb
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH -w kg35-nvl01
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --output /home/songs1/slurm-logs/pclr-heedb-%j.out

set -e

PRETRAIN_DATASET=/opt/gpudata/ecg/heedb
RUN_DIR=/opt/gpu_working/steven/protossl-ecg-outputs-rebuttal
REPO_ROOT=/opt/gpu_working/steven/ProtoSSL

# set these env vars prior to executing this script
: "${PRETRAIN_DATASET:?Env var PRETRAIN_DATASET must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"
echo "Using PRETRAIN_DATASET=$PRETRAIN_DATASET"
echo "Using RUN_DIR=$RUN_DIR"
echo "Using REPO_ROOT=$REPO_ROOT"
cd $REPO_ROOT/scripts

# submit job with 500GB of memory
export HIGH_MEMORY=1

# experiment parameters
EXP_NAME="pclr-heedb"
BACKBONE=resnet18
CONV=2D

# pretrain via self supervised contrastive learning
python -m protossl.trainer \
    --pipeline-stage train-contraster \
    --config $REPO_ROOT/configs/pretrain-unsupervised-no-proto.yaml \
    --model.backbone_type $BACKBONE \
    --model.conv_type $CONV \
    --trainer.max_epochs 100 \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $PRETRAIN_DATASET
