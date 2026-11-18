#!/usr/bin/env bash
# Logistic regression on EchoNext's tabular features, unweighted and class-balanced.
#   slurm/submit.sh -- scripts/echonext/run-tabular-logreg.sh <dataset> <run> <seed>
#   <dataset>: echonext or a subset, e.g. echonext-256 (<seed> only names the output dir)
set -euo pipefail
source "$(dirname "$0")/../_common.sh" "$@"

EXP_NAME=tabular-logreg

stage "$EXP_NAME-unweighted" "$EXP_NAME-weighted"

python echonext/_tabular_logreg.py \
--echonext-data $DATASET_PATH \
--output-path $RUN_DIR/$EXP_NAME-unweighted
python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME-unweighted/probs.npy \
--output-path $RUN_DIR/$EXP_NAME-unweighted
publish "$EXP_NAME-unweighted"

python echonext/_tabular_logreg.py \
--echonext-data $DATASET_PATH \
--balance-class-weight \
--output-path $RUN_DIR/$EXP_NAME-weighted
python _eval_probs.py \
--dataset-path $DATASET_PATH \
--probs-npy $RUN_DIR/$EXP_NAME-weighted/probs.npy \
--output-path $RUN_DIR/$EXP_NAME-weighted
publish "$EXP_NAME-weighted"
