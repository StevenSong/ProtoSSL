#!/usr/bin/env bash
#   slurm/submit.sh --gres=gpu:1 --mem-per-gpu=500G -- scripts/pretrain/run-pclr-heedb-pretrain.sh <run>
# see scripts/pretrain/_common.sh for where outputs go
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

# load each split into memory (hence --mem-per-gpu=500G)
export HIGH_MEMORY=1

# experiment parameters
EXP_NAME="pclr-heedb"
BACKBONE=resnet18
CONV=2D

stage "$EXP_NAME"

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

publish "$EXP_NAME"
