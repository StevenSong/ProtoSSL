#!/bin/bash

set -e

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
echo "Using DATASET_PATH=$DATASET_PATH"
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm
cd $REPO_ROOT/scripts

python _protoecgnet_cache_ptbxl.py --ptbxl-data $DATASET_PATH
python _pass_pclr_cache_echonext.py --ptbxl-data $DATASET_PATH
