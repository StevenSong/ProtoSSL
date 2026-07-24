#!/bin/bash

#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=500gb
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH -w kg35-nvl02
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --output /home/songs1/slurm-logs/supproto-heedb-150-%j.out

set -e

PRETRAIN_DATASET=/opt/gpudata/ecg/heedb
RUN_DIR=/home/songs1/protossl-ecg-outputs-rebuttal
REPO_ROOT=/home/songs1/ProtoSSL

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

# toggle HEEDB label set
export USE_HEEDB_150=1

# experiment parameters
EXP_NAME="supproto-heedb-150"
BACKBONE=resnet18
CONV=2D

# pretrain via label supervised prototype learning
python -m protossl.trainer \
    --pipeline-stage learn-prototypes-supervised \
    --config $REPO_ROOT/configs/pretrain-supervised-heedb-150.yaml \
    --model.backbone_type $BACKBONE \
    --model.conv_type $CONV \
    --trainer.max_epochs 100 \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $PRETRAIN_DATASET \
    --data.num_workers 8 \
    --data.prefetch_factor 4

# project in the pretraining dataset
python -m protossl.trainer \
    --pipeline-stage project-prototypes-supervised \
    --config $REPO_ROOT/configs/pretrain-supervised-heedb-150.yaml \
    --model.backbone_type $BACKBONE \
    --model.conv_type $CONV \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $PRETRAIN_DATASET \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototypes-supervised/latest/best.ckpt \
    --data.num_workers 8 \
    --data.prefetch_factor 4

python -m protossl.trainer \
    --pipeline-stage train-classifier \
    --config $REPO_ROOT/configs/pretrain-supervised-heedb-150.yaml \
    --model.backbone_type $BACKBONE \
    --model.conv_type $CONV \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $PRETRAIN_DATASET \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes-supervised/latest/proj.ckpt \
    --data.num_workers 8 \
    --data.prefetch_factor 4

python _eval_probs.py \
--dataset-path $PRETRAIN_DATASET \
--probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

cp $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy $RUN_DIR/$EXP_NAME/probs.npy
