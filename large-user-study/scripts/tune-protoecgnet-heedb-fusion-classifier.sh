#!/bin/bash

#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=200gb
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

# experiment parameters
EXP_NAME="fusion-tuned"
# space separated branch names to ablate out of the head, empty keeps all of them.
# the cached embeddings always hold every branch, so this needs no re-embedding -
# but each setting gets its own output dir (and so its own optuna study)
DROP_BRANCHES="2D-global"
DROP_SUFFIX=${DROP_BRANCHES:+-drop-${DROP_BRANCHES// /-}}
EMBEDS_DIR=$RUN_DIR/$EXP_NAME/compute-fusion-embeddings/latest
OUTPUT_DIR=$RUN_DIR/$EXP_NAME/tune-fusion-classifier$DROP_SUFFIX

# re-running resumes the optuna study (sqlite in $OUTPUT_DIR) up to --n-trials total
# use --export-only --ap-tolerance X to reselect without retraining
python _tune_fusion_classifier.py \
    --embeds-dir $EMBEDS_DIR \
    --output-dir $OUTPUT_DIR \
    ${DROP_BRANCHES:+--drop-branches $DROP_BRANCHES} \
    --n-trials 50

# labels.yaml is written by the tuning script: dropping a branch can drop labels,
# so the eval has to score the same label subset the head was tuned on
python _eval_probs_bootstrapped.py \
--dataset-path $DATASET_PATH \
--data-kwargs '{"heedb_split_type": "by-label"}' \
--label-subset-config $OUTPUT_DIR/labels.yaml \
--probs-npy $OUTPUT_DIR/test_probs.npy \
--output-path $OUTPUT_DIR

# suffixed so an ablation does not clobber the all-branches probs
cp $OUTPUT_DIR/test_probs.npy $RUN_DIR/$EXP_NAME/probs$DROP_SUFFIX.npy
