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

export DATASET_PATH=/opt/gpudata/audio/VoxCeleb1
export RUN_DIR=/opt/gpu_working/steven/protossl-audio/runs-voxceleb
export PPL=2ppl

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"
: "${SEED:=42}"
: "${PPL:=5ppl}"
echo "Using PPL=$PPL"
echo "Using SEED=$SEED"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
echo "Using REPO_ROOT=$REPO_ROOT"
cd $REPO_ROOT/scripts/audio

# experiment parameters
EXP_NAME="protossl-audioset-pila"
PRETRAIN_RUN="$RUN_DIR/../pass-audioset"

# python -m protossl.trainer \
#     --config $REPO_ROOT/configs/audio/target-guided-$PPL.yaml \
#     --seed_everything $SEED \
#     --pipeline-stage learn-prototype-assignments \
#     --assignment-strategy ilp_effect_size \
#     --model.n_prototypes 2635 \
#     --trainer.logger.save_dir $RUN_DIR \
#     --trainer.logger.name $EXP_NAME \
#     --data.dataset_path $DATASET_PATH \
#     --model.pretrained_weights $PRETRAIN_RUN/learn-prototypes/latest/best.ckpt \
#     --model.model_kwargs '{"label_type": "multiclass"}'

# python -m protossl.trainer \
#     --config $REPO_ROOT/configs/audio/target-guided-$PPL.yaml \
#     --seed_everything $SEED \
#     --pipeline-stage project-prototypes-supervised \
#     --trainer.logger.save_dir $RUN_DIR \
#     --trainer.logger.name $EXP_NAME \
#     --data.dataset_path $DATASET_PATH \
#     --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototype-assignments/latest/assigned.ckpt

# python -m protossl.trainer \
#     --config $REPO_ROOT/configs/audio/target-guided-$PPL.yaml \
#     --seed_everything $SEED \
#     --pipeline-stage train-classifier \
#     --trainer.logger.save_dir $RUN_DIR \
#     --trainer.logger.name $EXP_NAME \
#     --data.dataset_path $DATASET_PATH \
#     --model.model_kwargs '{"label_type": "multiclass"}' \
#     --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes-supervised/latest/proj.ckpt

# cp $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy $RUN_DIR/$EXP_NAME/probs.npy

# python _eval_probs.py \
# --dataset-path $DATASET_PATH \
# --probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
# --output-path $RUN_DIR/$EXP_NAME

python _eval_probs_bootstrapped.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

# now fine-tune
PRETRAIN_RUN=$RUN_DIR/$EXP_NAME
EXP_NAME="$EXP_NAME-ft"

# python -m protossl.trainer \
#     --config $REPO_ROOT/configs/audio/target-guided-$PPL.yaml \
#     --seed_everything $SEED \
#     --pipeline-stage learn-prototypes-supervised \
#     --trainer.logger.save_dir $RUN_DIR/ \
#     --trainer.logger.name $EXP_NAME \
#     --data.dataset_path $DATASET_PATH \
#     --model.pretrained_weights $PRETRAIN_RUN/learn-prototype-assignments/latest/assigned.ckpt \
#     --model.model_kwargs '{"label_type": "multiclass", "use_default_weights": True}'

# python -m protossl.trainer \
#     --config $REPO_ROOT/configs/audio/target-guided-$PPL.yaml \
#     --seed_everything $SEED \
#     --pipeline-stage project-prototypes-supervised \
#     --trainer.logger.save_dir $RUN_DIR/ \
#     --trainer.logger.name $EXP_NAME \
#     --data.dataset_path $DATASET_PATH \
#     --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototypes-supervised/latest/best.ckpt

# python -m protossl.trainer \
#     --config $REPO_ROOT/configs/audio/target-guided-$PPL.yaml \
#     --seed_everything $SEED \
#     --pipeline-stage train-classifier \
#     --trainer.logger.save_dir $RUN_DIR \
#     --trainer.logger.name $EXP_NAME \
#     --data.dataset_path $DATASET_PATH \
#     --model.model_kwargs '{"label_type": "multiclass"}' \
#     --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes-supervised/latest/proj.ckpt

# cp $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy $RUN_DIR/$EXP_NAME/probs.npy

# python _eval_probs.py \
# --dataset-path $DATASET_PATH \
# --probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
# --output-path $RUN_DIR/$EXP_NAME

python _eval_probs_bootstrapped.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME/probs.npy \
--output-path $RUN_DIR/$EXP_NAME
