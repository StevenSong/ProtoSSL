# this file should be sourced by the queue scripts, so they can use `submit_job`

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "THIS FILE (_submit_job.sh) SHOULD NOT BE EXECUTED DIRECTLY"
    exit 1
fi

function submit_job() {
    # $BASE_DATASET_PATH - base path upon which suffix is appended (do not use trailing slash)
    # $BASE_RUN_DIR - base path upon which suffix is appended (do not use trailing slash)
    # $1 - suffix
    # $2 - script
    # $3 - custom flags
    : "${BASE_DATASET_PATH:?Env var BASE_DATASET_PATH must be set prior to script execution}"
    : "${BASE_RUN_DIR:?Env var BASE_RUN_DIR must be set prior to script execution}"
    : "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"

    local fname="${2%.sh}"

    local job_id=$(sbatch -p a100 --gpus=1 \
    --time=0 --cpus-per-task=12 --mem=200g --ntasks=1 \
    --export=ALL,DATASET_PATH=$BASE_DATASET_PATH$1,RUN_DIR=$BASE_RUN_DIR$1,REPO_ROOT=$REPO_ROOT \
    --output slurm-logs/$fname$1-%j.out $3 --parsable _slurm_wrapper.sh $2)

    echo $job_id
}