#!/bin/bash

set -e

# set these env vars prior to executing this script
: "${DATASET_PATH:?Env var DATASET_PATH must be set prior to script execution}"
: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"
echo "Using DATASET_PATH=$DATASET_PATH"
echo "Using REPO_ROOT=$REPO_ROOT"
cd $REPO_ROOT/scripts

python _cache_data.py --dataset-path $DATASET_PATH
