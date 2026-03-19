#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40gb
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpuq
#SBATCH --time=00-23:59:59

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
EXP_NAME="htsat"

python -m pass_pclr.trainer \
    --config $REPO_ROOT/configs/resnet.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --data.batch_size 8 \
    --data.sampling_rate 32000 \
    --model.conv_type HTSAT \
    --model.input_channels 1 \
    --model.resnet_type resnet18 \
    --model.audio_backbone_name Cnn14

python _eval_probs.py \
    --dataset-path $DATASET_PATH \
    --probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
    --output-path $RUN_DIR/$EXP_NAME

cp $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy $RUN_DIR/$EXP_NAME/probs.npy