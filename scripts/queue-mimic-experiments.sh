#!/bin/bash

set -e

SUFFIXES=("" "-32k" "-16k" "-8k" "-4k" "-2k" "-1k" "-512" "-256")

export BASE_DATASET_PATH=/opt/gpudata/ecg/mimic-iv-ecg
export BASE_RUN_DIR=/opt/gpudata/steven/ecg-prototype-fm/outputs/runs-mimic
export REPO_ROOT=/opt/gpudata/steven/ecg-prototype-fm

source _submit_job.sh

for suffix in "${SUFFIXES[@]}"; do
    echo "SUFFIX: $suffix"

    cache_id=$(submit_job "$suffix" 0-run-cache-data.sh)
    echo $cache_id

    pfs_id=$(submit_job "$suffix" 1-1-run-proto-from-scratch.sh "--dependency=afterok:$cache_id")
    echo $pfs_id
    submit_job "$suffix" 1-2-run-proto-from-scratch-logreg.sh "--dependency=afterok:$cache_id,$pfs_id"

    submit_job "$suffix" 2-1-run-pass-heedb-pip.sh "--dependency=afterok:$cache_id"
    pit_id=$(submit_job "$suffix" 2-2-run-pass-heedb-pit.sh "--dependency=afterok:$cache_id")
    echo $pit_id

    submit_job "$suffix" 2-3-run-pass-heedb-pip-logreg.sh "--dependency=afterok:$cache_id"
    submit_job "$suffix" 2-4-run-pass-heedb-pit-logreg.sh "--dependency=afterok:$cache_id,$pit_id"

    pit_assign_id=$(submit_job "$suffix" 2-5-run-pass-heedb-pit-assign.sh "--dependency=afterok:$cache_id")
    echo $pit_assign_id

    submit_job "$suffix" 2-6-run-pass-heedb-pit-assign-logreg.sh "--dependency=afterok:$cache_id,$pit_assign_id"

    submit_job "$suffix" 3-1-run-prosup-heedb-pip.sh "--dependency=afterok:$cache_id"
    pit_id=$(submit_job "$suffix" 3-2-run-prosup-heedb-pit.sh "--dependency=afterok:$cache_id")
    echo $pit_id

    submit_job "$suffix" 3-3-run-prosup-heedb-pip-logreg.sh "--dependency=afterok:$cache_id"
    submit_job "$suffix" 3-4-run-prosup-heedb-pit-logreg.sh "--dependency=afterok:$cache_id,$pit_id"

    pit_assign_id=$(submit_job "$suffix" 3-5-run-prosup-heedb-pit-assign.sh "--dependency=afterok:$cache_id")
    echo $pit_assign_id

    submit_job "$suffix" 3-6-run-prosup-heedb-pit-assign-logreg.sh "--dependency=afterok:$cache_id,$pit_assign_id"

    pit_id=$(submit_job "$suffix" 3-7-run-prosup-heedb-pip-then-pit.sh "--dependency=afterok:$cache_id")
    echo $pit_id
    submit_job "$suffix" 3-8-run-prosup-heedb-pip-then-pit-logreg.sh "--dependency=afterok:$cache_id,$pit_id"
    pit_assign_id=$(submit_job "$suffix" 3-9-run-prosup-heedb-pip-then-pit-assign.sh "--dependency=afterok:$cache_id")
    echo $pit_assign_id
    submit_job "$suffix" 3-10-run-prosup-heedb-pip-then-pit-assign-logreg.sh "--dependency=afterok:$cache_id,$pit_assign_id"
done
