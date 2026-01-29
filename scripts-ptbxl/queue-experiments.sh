#!/bin/bash

set -e

SUFFIXES=("" "-8k" "-4k" "-2k")

function submit_job() {
    # $1 - suffix
    # $2 - script
    # $3 - custom flags

    local fname="${2%.sh}"

    local job_id=$(sbatch -p a100 --gpus=1 \
    --time=0 --cpus-per-task=12 --mem=200g --ntasks=1 \
    --export=ALL,DATASET_PATH=/opt/gpudata/ecg/ptb-xl$1,RUN_DIR=/opt/gpudata/steven/ecg-prototype-fm/outputs/runs-ptbxl$1 \
    --output slurm-logs/$fname$1-%j.out $3 --parsable _slurm_wrapper.sh $2)

    echo $job_id
}


for suffix in "${SUFFIXES[@]}"; do
    echo "SUFFIX: $suffix"

    # cache the transformed data
    cache_id=$(submit_job "$suffix" 0-0-run-cache-ptbxl-data.sh)
    echo $cache_id

    # submit all jobs
    cooc_id=$(submit_job "$suffix" 5-0-run-protoecgnet-echonext-cooccurrence.sh "--dependency=afterok:$cache_id")
    echo $cooc_id
    submit_job "$suffix" 5-1-run-protoecgnet-echonext-cat1.sh "--dependency=afterok:$cache_id,$cooc_id"

    submit_job "$suffix" 8-2-run-pass-pclr-transfer-heedb.sh "--dependency=afterok:$cache_id"
    with_proj_id=$(submit_job "$suffix" 8-3-run-pass-pclr-transfer-heedb-with-proj.sh "--dependency=afterok:$cache_id")
    echo $with_proj_id

    submit_job "$suffix" 8-4-run-pass-pclr-transfer-heedb-logreg.sh "--dependency=afterok:$cache_id"
    submit_job "$suffix" 8-5-run-pass-pclr-transfer-heedb-with-proj-logreg.sh "--dependency=afterok:$cache_id,$with_proj_id"
done
