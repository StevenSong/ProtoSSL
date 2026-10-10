#!/bin/bash

set -e

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"
: "${SEED:=42}"
echo "Using SEED=$SEED"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
echo "Using REPO_ROOT=$REPO_ROOT"
cd $REPO_ROOT/scripts

# experiment parameters
EXP_NAME="protossl-heedb-150-pila"
PRETRAIN_RUN=$(realpath -m "$RUN_DIR/../../protossl-heedb-150")

python -m protossl.trainer \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --seed_everything $SEED \
    --pipeline-stage learn-prototype-assignments \
    --assignment-strategy ilp_effect_size \
    --model.n_prototypes 7500 \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $PRETRAIN_RUN/learn-prototypes/latest/best.ckpt

python -m protossl.trainer \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --seed_everything $SEED \
    --pipeline-stage project-prototypes-supervised \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototype-assignments/latest/assigned.ckpt

python -m protossl.trainer \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --seed_everything $SEED \
    --pipeline-stage compute-embeddings \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes-supervised/latest/proj.ckpt

python _linear_probe.py \
--random-seed $SEED \
--dataset-path $DATASET_PATH \
--prototype-embeddings $RUN_DIR/$EXP_NAME/compute-embeddings/latest \
--output-path $RUN_DIR/$EXP_NAME

python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

python _eval_probs_bootstrapped.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

# now fine-tune
PRETRAIN_RUN=$RUN_DIR/$EXP_NAME
EXP_NAME="$EXP_NAME-ft"

python -m protossl.trainer \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --seed_everything $SEED \
    --pipeline-stage learn-prototypes-supervised \
    --trainer.logger.save_dir $RUN_DIR/ \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $PRETRAIN_RUN/learn-prototype-assignments/latest/assigned.ckpt

python -m protossl.trainer \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --seed_everything $SEED \
    --pipeline-stage project-prototypes-supervised \
    --trainer.logger.save_dir $RUN_DIR/ \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototypes-supervised/latest/best.ckpt

python -m protossl.trainer \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --seed_everything $SEED \
    --pipeline-stage compute-embeddings \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes-supervised/latest/proj.ckpt

python _linear_probe.py \
--random-seed $SEED \
--dataset-path $DATASET_PATH \
--prototype-embeddings $RUN_DIR/$EXP_NAME/compute-embeddings/latest \
--output-path $RUN_DIR/$EXP_NAME

python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

python _eval_probs_bootstrapped.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
--output-path $RUN_DIR/$EXP_NAME
