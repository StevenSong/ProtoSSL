#!/bin/bash

set -e

HEEDB_DATA=/opt/gpudata/ecg/heedb
RUN_DIR=/opt/gpu_working/steven/ProtoSSL/data-preprocessing/heedb_normalizations

python _compute_heedb_normalizations.py --dataset_path $HEEDB_DATA --output_path $RUN_DIR
