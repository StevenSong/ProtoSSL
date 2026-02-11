#!/bin/bash

set -e

HEEDB_DATA=/opt/gpudata/ecg/heedb
RUN_DIR=/opt/gpudata/steven/ecg-prototype-fm/outputs/heedb_normalizations
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm
cd $REPO_ROOT/scripts/echonext

python _compute_heedb_normalizations.py --dataset_path $HEEDB_DATA --output_path $RUN_DIR
