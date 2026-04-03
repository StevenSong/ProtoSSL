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

# ==============================================================================
# Learn Assignments (ProtoPool)
# ==============================================================================
PRETRAIN_RUN="$RUN_DIR/../prosup-pretrain-heedb"
EXP_NAME="labsup-proto-heedb-ria"

python -m pass_pclr.trainer \
    --pipeline-stage learn-prototype-assignments \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --assignment-strategy protopool \
    --model.n_prototypes 1000 \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $PRETRAIN_RUN/learn-prototypes/latest/best.ckpt

python -m pass_pclr.trainer \
    --pipeline-stage project-prototypes-supervised \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototype-assignments/latest/assigned.ckpt

python -m pass_pclr.trainer \
    --pipeline-stage train-classifier \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --model.extra_kwargs '{"use_regularization_mask": True, "use_proto_cls_init": True}' \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes-supervised/latest/proj.ckpt

python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

# ==============================================================================
# Logistic Regregression
# ==============================================================================
PRETRAIN_RUN="$RUN_DIR/$EXP_NAME"
EXP_NAME_LR="$EXP_NAME-lr"

python -m pass_pclr.trainer \
    --pipeline-stage compute-embeddings \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME_LR \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $PRETRAIN_RUN/project-prototypes-supervised/latest/proj.ckpt

python _linear_probe.py \
--dataset-path $DATASET_PATH \
--prototype-embeddings $RUN_DIR/$EXP_NAME_LR/compute-embeddings/latest \
--output-path $RUN_DIR/$EXP_NAME_LR

python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME_LR/probs.npy \
--output-path $RUN_DIR/$EXP_NAME_LR

# ==============================================================================
# Fine Tune
# ==============================================================================
PRETRAIN_RUN="$RUN_DIR/$EXP_NAME"
EXP_NAME_FT="$EXP_NAME-ft"

python -m pass_pclr.trainer \
    --pipeline-stage learn-prototypes-supervised \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --model.extra_kwargs '{"use_regularization_mask": True, "use_proto_cls_init": False}' \
    --model.do_finetune True \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME_FT \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $PRETRAIN_RUN/train-classifier/latest/best.ckpt

python -m pass_pclr.trainer \
    --pipeline-stage project-prototypes-supervised \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME_FT \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME_FT/learn-prototypes-supervised/latest/best.ckpt

python -m pass_pclr.trainer \
    --pipeline-stage train-classifier \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --model.extra_kwargs '{"use_regularization_mask": True, "use_proto_cls_init": True}' \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME_FT \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME_FT/project-prototypes-supervised/latest/proj.ckpt

python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME_FT/train-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME_FT

# ==============================================================================
# Fine Tuned Logistic Regression
# ==============================================================================
PRETRAIN_RUN="$RUN_DIR/$EXP_NAME-ft"
EXP_NAME_FT_LR="$EXP_NAME-ft-lr"

python -m pass_pclr.trainer \
    --pipeline-stage compute-embeddings \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME_FT_LR \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $PRETRAIN_RUN/project-prototypes-supervised/latest/proj.ckpt

python _linear_probe.py \
--dataset-path $DATASET_PATH \
--prototype-embeddings $RUN_DIR/$EXP_NAME_FT_LR/compute-embeddings/latest \
--output-path $RUN_DIR/$EXP_NAME_FT_LR

python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME_FT_LR/probs.npy \
--output-path $RUN_DIR/$EXP_NAME_FT_LR
