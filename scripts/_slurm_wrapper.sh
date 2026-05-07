#!/bin/bash

# assumes submitting context has exported CONDA_EXE
source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
conda activate protossl
# echo WOULD RUN:
# echo $1
# echo "SEED=$SEED"
# echo "DATASET_PATH=$DATASET_PATH"
# echo "RUN_DIR=$RUN_DIR"

# Override python to use srun when submitted via this wrapper
python() { srun -u python "$@"; }
python3() { srun -u python3 "$@"; }
export -f python
export -f python3

case "$1" in
    *.sh) bash "$1" "${@:2}" ;;
    *.py) python3 "$1" "${@:2}" ;;
    *)    echo "Unsupported extension" >&2; exit 1 ;;
esac
