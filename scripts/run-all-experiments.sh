#!/bin/bash

set -e

# SUFFIXES=("" "-32k" "-16k" "-8k" "-4k" "-2k") # already ran over full dataset (i.e. "")
SUFFIXES=("-32k" "-16k" "-8k" "-4k" "-2k")

function submit_job() {
    # $1 - suffix
    # $2 - script
    # $3 - dependent job id [optional]

    if [[ -n "$3" ]]; then
        dependency="--dependency=afterok:$3"
    fi

    fname="${2%.sh}"

    job_id=$(sbatch -p gpu --gpus=1  --nodelist kl35-gpu-5 \
    --time=0 --cpus-per-task=12 --mem=200g --ntasks=1 \
    --export=ALL,ECHONEXT_DATA=/opt/gpudata/ecg/echonext$1,RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/runs$1 \
    --output slurm-logs/$fname$1-%j.out $dependency --parsable $2)

    echo $job_id
}


for suffix in "${SUFFIXES[@]}"; do
    submit_job $suffix 1-run-echonext-logreg.sh
    submit_job $suffix 2-run-minimodel.sh
    submit_job $suffix 3-run-resnet-1d.sh
    submit_job $suffix 3-run-resnet-2d.sh
    submit_job $suffix 4-0-run-protoecgnet-probe-sklearn.sh
    submit_job $suffix 4-1-run-protoecgnet-probe-torch-cat1.sh
    submit_job $suffix 4-3-run-protoecgnet-probe-torch-cat3.sh
    submit_job $suffix 4-4-run-protoecgnet-probe-torch-cat4.sh
    submit_job $suffix 4-5-run-protoecgnet-probe-torch-fusion.sh
    cooc_id=$(submit_job $suffix 5-0-run-protoecgnet-echonext-cooccurrence.sh)
    cat1_id=$(submit_job $suffix 5-1-run-protoecgnet-echonext-cat1.sh $cooc_id)
    cat3_id=$(submit_job $suffix 5-3-run-protoecgnet-echonext-cat3.sh $cooc_id)
    cat4_id=$(submit_job $suffix 5-4-run-protoecgnet-echonext-cat4.sh $cooc_id)
    submit_job $suffix 5-5-run-protoecgnet-echonext-fusion.sh "$cat1_id,$cat3_id,$cat4_id"
done
