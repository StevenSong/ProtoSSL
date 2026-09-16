#!/bin/bash

#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=300gb
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
BRANCH="2D-partial"
EXP_NAME="$BRANCH-tuned"
# best lambda-sweep trial (wandb sweep blqwqt64, run btq9wfqt, val_ap=0.5352)
TUNED_TRIAL="v294-895m7c3p"

# project in the training dataset
python -m protossl.protoecgnet_trainer \
    --pipeline-stage project-prototypes-supervised \
    --config $REPO_ROOT/configs/$BRANCH.yaml \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --data.data_kwargs '{"heedb_split_type": "by-label"}' \
    --model.pretrained_weights $RUN_DIR/$EXP_NAME/learn-prototypes-supervised/$TUNED_TRIAL/best.ckpt \
    --data.num_workers 8 \
    --data.prefetch_factor 4
