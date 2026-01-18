#!/bin/bash

set -e

SUFFIXES=("" "-32k" "-16k" "-8k" "-4k" "-2k" "-1k" "-512" "-256")

function submit_job() {
    # $1 - suffix
    # $2 - script
    # $3 - custom flags

    local fname="${2%.sh}"

    local job_id=$(sbatch -p a100 --gpus=1 \
    --time=0 --cpus-per-task=12 --mem=200g --ntasks=1 \
    --export=ALL,ECHONEXT_DATA=/opt/gpudata/ecg/echonext$1,RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/outputs/runs$1 \
    --output slurm-logs/$fname$1-%j.out $3 --parsable _slurm_wrapper.sh $2)

    echo $job_id
}

# first do pretraining
# make sure that pretraining is not going to cache the same dataset as the transfer
pretrain_id=$(submit_job "" 7-1-run-pass-pclr-pretrain.sh)
echo $pretrain_id

# then do transfer
for suffix in "${SUFFIXES[@]}"; do
    echo "SUFFIX: $suffix"

    # cache the transformed data to prevent race conditions within the parallelized jobs
    cache_id=$(submit_job "$suffix" 7-0-run-pass-pclr-cache-data.sh)

    submit_job "$suffix" 7-2-run-pass-pclr-transfer.sh "--dependency=afterok:$pretrain_id,$cache_id"
    submit_job "$suffix" 7-3-run-pass-pclr-transfer-with-proj.sh "--dependency=afterok:$pretrain_id,$cache_id"
done
