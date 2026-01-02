#!/bin/bash

source /opt/gpudata/steven/miniforge3/etc/profile.d/conda.sh
conda activate ecg
# echo WOULD RUN:
# echo $1
bash $1
