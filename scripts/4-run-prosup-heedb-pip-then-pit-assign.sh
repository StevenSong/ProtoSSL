#!/bin/bash

set -e

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
echo "Using REPO_ROOT=$REPO_ROOT"
cd $REPO_ROOT/scripts

# experiment parameters
EXP_NAME="prosup-heedb-pip-then-pit-assign"
PRETRAIN_RUN="/opt/gpudata/steven/ecg-prototype-fm/outputs/prosup-pretrain-heedb"
PPL=5

# this version relies on learning prototype assignments relative to the target task
python -m pass_pclr.trainer \
    --pipeline-stage learn-prototype-assignments \
    --config $REPO_ROOT/configs/pass-pclr.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $PRETRAIN_RUN/project-prototypes-supervised/latest/proj.ckpt \
    --model.n_prototypes_per_label $PPL \
    --model.n_prototypes 100  # from 5 per label for 20 heedb labels

# then project
python -m pass_pclr.trainer \
    --pipeline-stage project-prototypes-supervised \
    --config $REPO_ROOT/configs/pass-pclr.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototype-assignments/latest/assigned.ckpt \
    --model.n_prototypes_per_label $PPL \
    --model.n_prototypes null

# then train classifier
python -m pass_pclr.trainer \
    --pipeline-stage train-classifier \
    --config $REPO_ROOT/configs/pass-pclr.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes-supervised/latest/proj.ckpt \
    --model.n_prototypes_per_label $PPL \
    --model.n_prototypes null

python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

cp $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy $RUN_DIR/$EXP_NAME/probs.npy

# now do logreg
PRETRAIN_RUN="$RUN_DIR/$EXP_NAME"
EXP_NAME="$EXP_NAME-logreg"

# this version relies on samples projected in the transfer dataset
python -m pass_pclr.trainer \
    --pipeline-stage compute-embeddings \
    --config $REPO_ROOT/configs/pass-pclr.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $PRETRAIN_RUN/project-prototypes-supervised/latest/proj.ckpt \
    --model.n_prototypes_per_label $PPL \
    --model.n_prototypes null

python _linear_probe.py \
--dataset-path $DATASET_PATH \
--prototype-embeddings $RUN_DIR/$EXP_NAME/compute-embeddings/latest \
--output-path $RUN_DIR/$EXP_NAME

python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
--output-path $RUN_DIR/$EXP_NAME
