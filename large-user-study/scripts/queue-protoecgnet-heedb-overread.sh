#!/bin/bash

set -e

# cache first so the arms don't race to build the same filtered caches
CACHE_JOB=$(sbatch --parsable run-cache-heedb-overread.sh)
echo "cache: $CACHE_JOB"

ARM_JOBS=()
for arm in 1D-global 2D-global 2D-partial; do
    job=$(sbatch --parsable --dependency=afterok:$CACHE_JOB run-protoecgnet-heedb-$arm.sh)
    echo "$arm: $job"
    ARM_JOBS+=($job)
done

deps=$(IFS=:; echo "${ARM_JOBS[*]}")
FUSION_JOB=$(sbatch --parsable --dependency=afterok:$deps run-protoecgnet-heedb-fusion.sh)
echo "fusion: $FUSION_JOB"
