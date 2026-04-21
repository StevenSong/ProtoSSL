#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40gb
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuq
#SBATCH --time=00-23:59:59

set -e

: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"

echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
echo "Using REPO_ROOT=$REPO_ROOT"

cd $REPO_ROOT/scripts

# experiment parameters
EXP_NAME="pass-audioset-assign"
PRETRAIN_RUN="/gpfs/data/bbj-lab/users/sethis/ecg-prototype-fm/results_audioset_proto2_contrastive/pass-audioset-ssl-assign/learn-prototypes/v2-f17up3vy/best.ckpt"

# contrastive pretraining mode for audio:
#   clar
#   cola
#   cola+clar
PAIR_MODE="cola+clar"
COLA_VIEW_SECONDS="2.0"
COLA_LOSS_WEIGHT="2.0"
CLAR_LOSS_WEIGHT="1.0"
KOLEO_LOSS_WEIGHT="1.0"

N_PROTOTYPES_PER_LABEL="5"
N_LABELS="527"
N_PROTOTYPES_SSL=$((N_PROTOTYPES_PER_LABEL * N_LABELS))

COMMON_ARGS=(
    --config $REPO_ROOT/configs/pass-pclr.yaml
    --trainer.logger.save_dir $RUN_DIR
    --trainer.logger.name $EXP_NAME
    --data.dataset_path $DATASET_PATH
    --data.batch_size 4
    --data.sampling_rate 32000
    --model.init_args.resnet_type resnet18
    --model.init_args.conv_type PANNS
    --model.init_args.input_channels 1
    --model.init_args.prototype_type partial
    --model.init_args.partial_len 32000
    --model.init_args.partial_overlap 0.5
    --model.init_args.prototype_h 1
    --model.init_args.prototype_w 1
    --model.init_args.audio_backbone_name Cnn14
)

echo "======================================"
echo "Stage 2: learn prototype assignments"
echo "======================================"

srun python -m pass_pclr.trainer \
    --pipeline-stage learn-prototype-assignments \
    --assignment-strategy ilp_effect_size \
    "${COMMON_ARGS[@]}" \
    --model.init_args.n_prototypes $N_PROTOTYPES_SSL \
    --model.init_args.n_prototypes_per_label $N_PROTOTYPES_PER_LABEL \
    --model.pretrained_weights $PRETRAIN_RUN

echo "======================================"
echo "Stage 3: project prototypes supervised"
echo "======================================"

srun python -m pass_pclr.trainer \
    --pipeline-stage project-prototypes-supervised \
    "${COMMON_ARGS[@]}" \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototype-assignments/latest/assigned.ckpt \
    --model.init_args.n_prototypes_per_label $N_PROTOTYPES_PER_LABEL \
    --model.init_args.n_prototypes null

echo "======================================"
echo "Stage 4: train classifier"
echo "======================================"

srun python -m pass_pclr.trainer \
    --pipeline-stage train-classifier \
    "${COMMON_ARGS[@]}" \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes-supervised/latest/proj.ckpt \
    --model.init_args.n_prototypes_per_label $N_PROTOTYPES_PER_LABEL \
    --model.init_args.n_prototypes null

echo "======================================"
echo "Stage 5: evaluate probabilities"
echo "======================================"

srun python _eval_probs.py \
    --dataset-path $DATASET_PATH \
    --probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
    --output-path $RUN_DIR/$EXP_NAME

cp $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy $RUN_DIR/$EXP_NAME/probs.npy