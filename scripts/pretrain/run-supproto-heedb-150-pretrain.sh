#!/usr/bin/env bash
#   slurm/submit.sh --gres=gpu:1 --mem-per-gpu=500G -- scripts/pretrain/run-supproto-heedb-150-pretrain.sh <run>
# see scripts/pretrain/_common.sh for where outputs go
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

# load each split into memory (hence --mem-per-gpu=500G)
export HIGH_MEMORY=1

# toggle HEEDB label set
export USE_HEEDB_150=1

# experiment parameters
EXP_NAME="supproto-heedb-150"
BACKBONE=resnet18
CONV=2D

stage "$EXP_NAME"

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

publish "$EXP_NAME"
