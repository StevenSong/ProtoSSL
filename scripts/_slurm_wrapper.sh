#!/bin/bash

# assumes submitting context has exported CONDA_EXE
source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
conda activate ecg
# echo WOULD RUN:
# echo $1
# echo "DATASET_PATH=$DATASET_PATH"
# echo "RUN_DIR=$RUN_DIR"

# Override python to use srun when submitted via this wrapper
python() { srun python "$@"; }
python3() { srun python3 "$@"; }
export -f python
export -f python3

bash $1
