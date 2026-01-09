#!/bin/bash

source /opt/gpudata/steven/miniforge3/etc/profile.d/conda.sh
conda activate ecg
# echo WOULD RUN:
# echo $1
# echo "ECHONEXT_DATA=$ECHONEXT_DATA"
# echo "RUN_DIR=$RUN_DIR"
bash $1
