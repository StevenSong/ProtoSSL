#!/bin/bash

#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=450gb
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH -w kg35-nvl01
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --output /home/songs1/slurm-logs/protoecgnet-heedb-%j.out

set -e

REPO_ROOT=/opt/gpu_working/steven/ProtoSSL/large-user-study

: "${REPO_ROOT:?Env var REPO_ROOT must be set prior to script execution}"
echo "Using REPO_ROOT=$REPO_ROOT"
cd $REPO_ROOT/scripts

python _cache_heedb_overread.py
