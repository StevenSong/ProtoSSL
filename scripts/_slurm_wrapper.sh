#!/bin/bash

# assumes submitting context has exported CONDA_PREFIX
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate ecg
# echo WOULD RUN:
# echo $1
# echo "DATASET_PATH=$DATASET_PATH"
# echo "RUN_DIR=$RUN_DIR"
bash $1
