#!/usr/bin/env bash
# Build the node-local 100 Hz dataset caches the experiments read (all splits).
#
# Optional: the first job on a node builds any missing cache itself, and the others on
# that node wait for it. Run this first to warm a node without holding a GPU meanwhile.
#   slurm/submit.sh -p h200 -w <node> -- scripts/0-run-cache-data.sh <dataset>
set -euo pipefail
(($# == 1)) || { echo "usage: $0 <dataset>" >&2; exit 1; }
cd "$(dirname "$0")"
source _datasets.sh
resolve_dataset "$1"

python _cache_data.py --dataset-path $DATASET_PATH
