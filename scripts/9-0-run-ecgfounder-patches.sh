#!/usr/bin/env bash
# ECGFounder patch embeddings for 9-1..9-3, which each run this first: built once per
# node and kept in the node-local stage (never published; several GB for full datasets).
#   scripts/9-0-run-ecgfounder-patches.sh <dataset> <run> <seed>
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

# experiment parameters
EXP_NAME="ecgfounder-patches"

mkdir -p "$RUN_DIR"
# 9-1..9-3 may run at once on a node: one builds, the others wait and reuse
exec 8>"$RUN_DIR/.$EXP_NAME.lock"
flock 8
if [[ -e $RUN_DIR/$EXP_NAME/.done ]]; then
  echo "reusing $RUN_DIR/$EXP_NAME"
  exit 0
fi
rm -rf "${RUN_DIR:?}/$EXP_NAME"

verify_sha256 "$ECGFOUNDER_CKPT" "$ECGFOUNDER_SHA256"
python ecg-fms/_compute_ecgfounder_embeddings.py \
--dataset-path $DATASET_PATH \
--output-path $RUN_DIR/$EXP_NAME \
--ckpt $ECGFOUNDER_CKPT \
--return-patches

touch "$RUN_DIR/$EXP_NAME/.done"
