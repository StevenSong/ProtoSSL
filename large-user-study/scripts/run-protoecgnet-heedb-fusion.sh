#!/bin/bash

#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=500gb
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH -w kg35-nvl01
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --output /home/songs1/slurm-logs/protoecgnet-heedb-%j.out

set -e

DATASET_PATH=/opt/gpudata/ecg/heedb
RUN_DIR=/opt/gpu_working/steven/protoecgnet-heedb
REPO_ROOT=/opt/gpu_working/steven/ProtoSSL/large-user-study

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${RUN_DIR:?Env var RUN_DIR must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using RUN_DIR=$RUN_DIR"
echo "Using REPO_ROOT=$REPO_ROOT"
cd $REPO_ROOT/scripts

# submit job with 500GB of memory
export HIGH_MEMORY=1

# experiment parameters
EXP_NAME="fusion"

python -m protossl.protoecgnet_trainer \
    --pipeline-stage train-fusion-classifier \
    --config $REPO_ROOT/configs/$EXP_NAME.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --data.data_kwargs '{"heedb_split_type": "by-label"}' \
    --model.branches+=protossl.models._protoecgnet.BranchCfg \
    --model.branches.name=1D-global \
    --model.branches.config=$REPO_ROOT/configs/1D-global.yaml \
    --model.branches.pretrained_weights=$RUN_DIR/1D-global/project-prototypes-supervised/latest/proj.ckpt \
    --model.branches+=protossl.models._protoecgnet.BranchCfg \
    --model.branches.name=2D-global \
    --model.branches.config=$REPO_ROOT/configs/2D-global.yaml \
    --model.branches.pretrained_weights=$RUN_DIR/2D-global/project-prototypes-supervised/latest/proj.ckpt \
    --model.branches+=protossl.models._protoecgnet.BranchCfg \
    --model.branches.name=2D-partial \
    --model.branches.config=$REPO_ROOT/configs/2D-partial.yaml \
    --model.branches.pretrained_weights=$RUN_DIR/2D-partial/project-prototypes-supervised/latest/proj.ckpt \
    --data.num_workers 8 \
    --data.prefetch_factor 4

python _eval_probs_bootstrapped.py \
--dataset-path $DATASET_PATH \
--data-kwargs '{"heedb_split_type": "by-label"}' \
--probs-npy $RUN_DIR/$EXP_NAME/train-fusion-classifier/latest/probs.npy \
--output-path $RUN_DIR/$EXP_NAME

cp $RUN_DIR/$EXP_NAME/train-fusion-classifier/latest/probs.npy $RUN_DIR/$EXP_NAME/probs.npy
