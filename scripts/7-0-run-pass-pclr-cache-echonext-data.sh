#!/bin/bash

set -e

# set these env vars prior to executing this script
# ECHONEXT_DATA=/opt/gpudata/ecg/echonext
: "${ECHONEXT_DATA:?Env var ECHONEXT_DATA must be set prior to script execution}"
echo "Using ECHONEXT_DATA=$ECHONEXT_DATA"
REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm
cd $REPO_ROOT/scripts

python _pass_pclr_cache_echonext.py --echonext-data $ECHONEXT_DATA
