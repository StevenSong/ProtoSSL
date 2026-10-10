#!/bin/bash

# PCLR (HEEDB-pretrained, no prototypes) linear probe

set -e

# declare -a DATASETS=("echonext" "ptbxl" "cinc" "mimic" "zzu" "code15")
declare -a DATASETS=("ptbxl" "cinc" "mimic" "zzu" "code15")
declare -a SEEDS=(42 67 70 73 99)
declare -A DATASET_DIRS=(
    ["echonext"]="echonext"
    ["ptbxl"]="ptb-xl"
    ["cinc"]="cinc-2020"
    ["mimic"]="mimic-iv-ecg"
    ["zzu"]="zzu-pecg"
    ["code15"]="code15"
)
declare -a SUFFIXES_echonext=("" "-32k" "-16k" "-8k" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_ptbxl=("" "-8k" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_code15=("" "-32k" "-16k" "-8k" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_cinc=("" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_mimic=("" "-32k" "-16k" "-8k" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_zzu=("" "-4k" "-2k" "-1k" "-512" "-256")

source _submit_job.sh

export BASE_ECG_PATH=/opt/gpudata/ecg
export REPO_ROOT=/opt/gpu_working/steven/ProtoSSL
# 5-3-run-pclr-logreg.sh finds the pretrained model at $RUN_DIR/../../pclr-heedb
BASE_OUTPUT_ROOT=/opt/gpu_working/steven/protossl-ecg-outputs-rebuttal

for seed in "${SEEDS[@]}"; do
    echo "SEED: $seed"
    export SEED=$seed
    export BASE_OUTPUT_DIR="$BASE_OUTPUT_ROOT/protossl-outputs-seed$SEED"

    for dataset in "${DATASETS[@]}"; do
        echo "DATASET: $dataset"
        ds_dir="${DATASET_DIRS[$dataset]}"
        export BASE_RUN_DIR="$BASE_OUTPUT_DIR/runs-$dataset"
        export BASE_DATASET_PATH="$BASE_ECG_PATH/$ds_dir"

        array_name="SUFFIXES_${dataset}[@]"
        suffixes=("${!array_name}")

        for suffix in "${suffixes[@]}"; do
            echo "SUFFIX: $suffix"
            # 100 Hz caches for all these datasets/sizes already exist from the main experiments
            submit_job "$suffix" 5-3-run-pclr-logreg.sh
        done
    done
done
