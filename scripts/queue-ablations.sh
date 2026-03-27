#!/bin/bash

set -e

declare -a DATASETS=("echonext" "ptbxl" "cinc" "mimic" "zzu")
declare -A DATASET_DIRS=(
    ["echonext"]="echonext"
    ["ptbxl"]="ptb-xl"
    ["cinc"]="cinc-2020"
    ["mimic"]="mimic-iv-ecg"
    ["zzu"]="zzu-pecg"
)
declare -a SUFFIXES_echonext=("" "-32k" "-16k" "-8k" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_ptbxl=("" "-8k" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_cinc=("" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_mimic=("" "-32k" "-16k" "-8k" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_zzu=("" "-4k" "-2k" "-1k" "-512" "-256")

export BASE_ECG_PATH=/opt/gpudata/ecg
export REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm
export BASE_OUTPUT_DIR=/opt/gpudata/steven/ecg-prototype-fm/outputs

source _submit_job.sh


for dataset in "${DATASETS[@]}"; do
    echo "DATASET: $dataset"
    ds_dir="${DATASET_DIRS[$dataset]}"
    export BASE_RUN_DIR="$BASE_OUTPUT_DIR/runs-$dataset"
    export BASE_DATASET_PATH="$BASE_ECG_PATH/$ds_dir"

    array_name="SUFFIXES_${dataset}[@]"
    suffixes=("${!array_name}")

    for suffix in "${suffixes[@]}"; do
        echo "SUFFIX: $suffix"

        cache_id=$(submit_job "$suffix" 0-run-cache-data.sh)
        echo $cache_id
        submit_job "$suffix" 5-01-run-pass-heedb-pip.sh "--dependency=afterok:$cache_id"
        submit_job "$suffix" 5-02-run-pass-heedb-pit.sh "--dependency=afterok:$cache_id"
        submit_job "$suffix" 5-03-run-prosup-heedb-pip.sh "--dependency=afterok:$cache_id"
        submit_job "$suffix" 5-04-run-prosup-heedb-pit.sh "--dependency=afterok:$cache_id"
        submit_job "$suffix" 5-05-run-prosup-heedb-pit-assign.sh "--dependency=afterok:$cache_id"
        submit_job "$suffix" 5-06-run-prosup-heedb-pip-then-pit.sh "--dependency=afterok:$cache_id"
        submit_job "$suffix" 5-07-run-prosup-heedb-pip-then-pit-assign.sh "--dependency=afterok:$cache_id"
    done
done