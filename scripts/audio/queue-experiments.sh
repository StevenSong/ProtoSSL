#!/bin/bash

scripts=(
    "1-run-blackbox-direct.sh"
    "2-run-labsup-proto-direct.sh"
    "3-run-protossl-audioset-pila.sh"
    "4-run-labsup-proto-audioset-rila.sh"
)

voxceleb_scripts=(
    "1-run-blackbox-direct.sh"
    "voxceleb-2-run-labsup-proto-direct.sh"
    "voxceleb-3-run-protossl-audioset-pila.sh"
    "voxceleb-4-run-labsup-proto-audioset-rila.sh"
)

# iemocap
for i in {0..4}; do
    export IEMOCAP_TEST_FOLD=$i
    export DATASET_PATH=/opt/gpudata/audio/IEMOCAP
    export RUN_DIR=/opt/gpu_working/steven/protossl-audio/runs-iemocap-fold$i
    for script in "${scripts[@]}"; do
        sbatch $script
    done
done

# urbansound8k
for i in {0..9}; do
    export US8K_TEST_FOLD=$i
    export DATASET_PATH=/opt/gpudata/audio/UrbanSound8K
    export RUN_DIR=/opt/gpu_working/steven/protossl-audio/runs-us8k-fold$i
    for script in "${scripts[@]}"; do
        sbatch $script
    done
done

# voxceleb1 (these take even longer to run)
export DATASET_PATH=/opt/gpudata/audio/VoxCeleb1
export RUN_DIR=/opt/gpu_working/steven/protossl-audio/runs-voxceleb
export PPL=2ppl
for script in "${voxceleb_scripts[@]}"; do
    sbatch $script
done
unset PPL
