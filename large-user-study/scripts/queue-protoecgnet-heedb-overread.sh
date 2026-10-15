#!/bin/bash

# usage: bash queue-protoecgnet-heedb-overread.sh [CACHE_JOB_ID]
# pass the id of an already submitted cache job to wait on it instead of submitting a new one

set -e

# cache first so the arms don't race to build the same filtered caches
CACHE_JOB=$1
if [ -z "$CACHE_JOB" ]; then
    CACHE_JOB=$(sbatch --parsable run-cache-heedb-overread.sh)
    echo "cache: $CACHE_JOB"
    DEPENDENCY="--dependency=afterok:$CACHE_JOB"
elif [ -n "$(squeue -h -j $CACHE_JOB 2>/dev/null)" ]; then
    echo "cache: $CACHE_JOB (existing, still queued/running)"
    DEPENDENCY="--dependency=afterok:$CACHE_JOB"
else
    # finished jobs get purged from the controller, so afterok can't be used on them
    state=$(sacct -n -X -j $CACHE_JOB -o State | tr -d ' ')
    if [ "$state" != "COMPLETED" ]; then
        echo "cache job $CACHE_JOB is not running and did not complete (state: '$state'), aborting"
        exit 1
    fi
    echo "cache: $CACHE_JOB (existing, already completed)"
    DEPENDENCY=""
fi

ARM_JOBS=()
for arm in 1D-global 2D-global 2D-partial; do
    job=$(sbatch --parsable $DEPENDENCY run-protoecgnet-heedb-$arm.sh)
    echo "$arm: $job"
    ARM_JOBS+=($job)
done

deps=$(IFS=:; echo "${ARM_JOBS[*]}")
FUSION_JOB=$(sbatch --parsable --dependency=afterok:$deps run-protoecgnet-heedb-fusion.sh)
echo "fusion: $FUSION_JOB"
