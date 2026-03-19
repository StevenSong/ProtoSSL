#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40gb
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
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
EXP_NAME="proto-from-scratch"

python -m pass_pclr.trainer \
    --pipeline-stage learn-prototypes-supervised \
    --config $REPO_ROOT/configs/proto-supervised.yaml \
    --trainer.logger.save_dir $RUN_DIR/ \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --data.batch_size 8 \
    --data.sampling_rate 32000 \
    --model.init_args.resnet_type resnet18 \
    --model.init_args.conv_type HTSAT \
    --model.init_args.input_channels 1 \
    --model.init_args.prototype_type partial \
    --model.init_args.partial_len 3200 \
    --model.init_args.partial_overlap 0.5 \
    --model.init_args.n_prototypes_per_label 5 \
    --model.init_args.prototype_h 2 \
    --model.init_args.prototype_w 2 


python -m pass_pclr.trainer \
    --pipeline-stage project-prototypes-supervised \
    --config $REPO_ROOT/configs/proto-supervised.yaml \
    --trainer.logger.save_dir $RUN_DIR/ \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --data.batch_size 8 \
    --data.sampling_rate 32000 \
    --model.init_args.resnet_type resnet18 \
    --model.init_args.conv_type HTSAT \
    --model.init_args.input_channels 1 \
    --model.init_args.prototype_type partial \
    --model.init_args.partial_len 3200 \
    --model.init_args.partial_overlap 0.5 \
    --model.init_args.n_prototypes_per_label 5 \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototypes-supervised/latest/best.ckpt \
    --model.init_args.prototype_h 2 \
    --model.init_args.prototype_w 2 

python -m pass_pclr.trainer \
    --pipeline-stage train-classifier \
    --config $REPO_ROOT/configs/proto-supervised.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --data.batch_size 8 \
    --data.sampling_rate 32000 \
    --model.init_args.resnet_type resnet18 \
    --model.init_args.conv_type HTSAT \
    --model.init_args.input_channels 1 \
    --model.init_args.prototype_type partial \
    --model.init_args.partial_len 3200 \
    --model.init_args.partial_overlap 0.5 \
    --model.init_args.n_prototypes_per_label 5 \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/project-prototypes-supervised/latest/proj.ckpt \
    --model.init_args.prototype_h 1 \
    --model.init_args.prototype_w 1 

python _eval_probs.py \
    --dataset-path $DATASET_PATH \
    --probs-npy $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy \
    --output-path $RUN_DIR/$EXP_NAME

cp $RUN_DIR/$EXP_NAME/train-classifier/latest/probs.npy $RUN_DIR/$EXP_NAME/probs.npy
