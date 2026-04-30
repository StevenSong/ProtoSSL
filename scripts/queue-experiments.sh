#!/bin/bash

set -e

declare -a DATASETS=("echonext" "ptbxl" "cinc" "mimic" "zzu" "code15")
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


for seed in "${SEEDS[@]}"; do
    echo "SEED: $seed"
    export SEED=$seed
    export BASE_OUTPUT_DIR=/opt/gpu_working/steven/new/protossl-outputs-seed"$SEED"

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

            submit_job "$suffix" 1-run-blackbox-direct.sh "--dependency=afterok:$cache_id"
            submit_job "$suffix" 2-run-labsup-proto-direct.sh "--dependency=afterok:$cache_id"
            submit_job "$suffix" 3-run-protossl-heedb-pila.sh "--dependency=afterok:$cache_id"
            submit_job "$suffix" 4-run-labsup-proto-heedb-rila.sh "--dependency=afterok:$cache_id"

            submit_job "$suffix" 5-1-run-ecgfounder-logreg.sh # does not depend on same 100 Hz cache (takes 500 Hz)
            submit_job "$suffix" 5-2-run-stmem-logreg.sh # does not depend on same 100 Hz cache (takes 250 Hz)

            if [ "$dataset" == "echonext" ]; then
                echo "Ablations"
                # ablation
                submit_job "$suffix" 6-run-protossl-heedb-pia.sh "--dependency=afterok:$cache_id"
                submit_job "$suffix" 7-run-protossl-heedb-pit.sh "--dependency=afterok:$cache_id"
                submit_job "$suffix" 8-run-protossl-heedb-pip.sh "--dependency=afterok:$cache_id"

                # start with pretrained encoder
                patches_id=$(submit_job "$suffix" 9-0-run-ecgfounder-patches.sh)
                echo $patches_id
                submit_job "$suffix" 9-1-run-ecgfounder-lap.sh "--dependency=afterok:$patches_id"
                submit_job "$suffix" 9-2-run-ecgfounder-clustering.sh "--dependency=afterok:$patches_id"
                submit_job "$suffix" 9-3-run-ecgfounder-random.sh "--dependency=afterok:$patches_id"
            fi
        done
    done
done
