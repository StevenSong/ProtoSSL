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
    --export=ALL,ECHONEXT_DATA=/opt/gpudata/ecg/echonext$1,RUN_DIR=/opt/gpudata/steven/ecg-prototype-fm/outputs/runs$1 \
    --output slurm-logs/$fname$1-%j.out $3 --parsable _slurm_wrapper.sh $2)

    echo $job_id
}


for suffix in "${SUFFIXES[@]}"; do

    # cache the transformed data
    hash=$(echo -n "/opt/gpudata/ecg/echonext$suffix" | md5sum | cut -c1-8)
    fname="../protoecgnet-cache/echonext_test_100hz_$hash.npy"
    if [ ! -f "$fname" ]; then
        echo caching echonext$suffix
        python _protoecgnet_cache_echonext.py --echonext-data /opt/gpudata/ecg/echonext$suffix
    fi

    echo "SUFFIX: $suffix"

    # submit all jobs
    submit_job "$suffix" 1-run-echonext-logreg.sh
    if [[ -z "$suffix" ]]; then
        # only run Columbia mini-model once, as we are not retraining it
        # over subsets of the training dataset (that's effectively the resnets)
        # NOTE if you do want to change this, beware that symlinked val/test
        # files within the echonext subset directories will break inside docker
        submit_job "$suffix" 2-run-minimodel.sh
    fi
    submit_job "$suffix" 3-run-resnet-1d.sh

    submit_job "$suffix" 4-0-run-protoecgnet-probe-sklearn.sh
    submit_job "$suffix" 4-1-run-protoecgnet-probe-torch-cat1.sh
    submit_job "$suffix" 4-3-run-protoecgnet-probe-torch-cat3.sh
    submit_job "$suffix" 4-4-run-protoecgnet-probe-torch-cat4.sh
    submit_job "$suffix" 4-5-run-protoecgnet-probe-torch-fusion.sh

    cooc_id=$(submit_job "$suffix" 5-0-run-protoecgnet-echonext-cooccurrence.sh)
    cat1_id=$(submit_job "$suffix" 5-1-run-protoecgnet-echonext-cat1.sh "--dependency=afterok:$cooc_id")
    cat3_id=$(submit_job "$suffix" 5-3-run-protoecgnet-echonext-cat3.sh "--dependency=afterok:$cooc_id")
    cat4_id=$(submit_job "$suffix" 5-4-run-protoecgnet-echonext-cat4.sh "--dependency=afterok:$cooc_id")
    submit_job "$suffix" 5-5-run-protoecgnet-echonext-fusion.sh "--dependency=afterok:$cat1_id,$cat3_id,$cat4_id"

    cat1_id=$(submit_job "$suffix" 6-1-run-protoecgnet-reproj-cat1.sh)
    cat3_id=$(submit_job "$suffix" 6-3-run-protoecgnet-reproj-cat3.sh)
    cat4_id=$(submit_job "$suffix" 6-4-run-protoecgnet-reproj-cat4.sh)
    submit_job "$suffix" 6-5-run-protoecgnet-reproj-fusion.sh "--dependency=afterok:$cat1_id,$cat3_id,$cat4_id"
done
