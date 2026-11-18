#!/usr/bin/env bash
# Queue the paper's downstream experiments for one run: every dataset, train size and
# seed. Run from a clone on a host with the Slurm client, at a pushed commit, once the
# pretrained models are published to the run by scripts/pretrain/ (or set
# PROTOSSL_PRETRAIN_RUN to read another run's; sbatch passes it on to the jobs).
#
#   scripts/queue-experiments.sh <run> [sbatch options for every job...]
#
# Each node a job lands on builds its own dataset caches on first use (see
# scripts/0-run-cache-data.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

usage="usage: scripts/queue-experiments.sh <run> [sbatch options...]"
run=${1:?$usage}
shift

declare -a DATASETS=("echonext" "ptbxl" "cinc" "mimic" "zzu" "code15")
declare -a SEEDS=(42 67 70 73 99)
declare -a SUFFIXES_echonext=("" "-32k" "-16k" "-8k" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_ptbxl=("" "-8k" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_code15=("" "-32k" "-16k" "-8k" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_cinc=("" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_mimic=("" "-32k" "-16k" "-8k" "-4k" "-2k" "-1k" "-512" "-256")
declare -a SUFFIXES_zzu=("" "-4k" "-2k" "-1k" "-512" "-256")

gpu=(--gres=gpu:1 --mem=200G "$@")
# q <script> <dataset> <seed>
q() {
  local job
  job=$(slurm/submit.sh --parsable "${gpu[@]}" -- "scripts/$1" "$2" "$run" "$3")
  echo "$1 $2 seed $3: $job"
}

for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    array_name="SUFFIXES_${dataset}[@]"
    for suffix in "${!array_name}"; do
      ds=$dataset$suffix

      q 1-run-blackbox-direct.sh "$ds" "$seed"
      q 2-run-supproto-direct.sh "$ds" "$seed"
      q 3-run-protossl-heedb-pila.sh "$ds" "$seed"
      q 4-run-supproto-heedb-rila.sh "$ds" "$seed"

      q 5-1-run-ecgfounder-logreg.sh "$ds" "$seed"
      q 5-2-run-stmem-logreg.sh "$ds" "$seed"
      q 5-3-run-pclr-logreg.sh "$ds" "$seed"

      if [[ $dataset == zzu && -z $suffix ]]; then
        # ZZU full scale, larger prototype bank
        q 3-run-protossl-heedb-150-pila.sh "$ds" "$seed"
      fi

      if [[ $dataset == echonext ]]; then
        # Ablations
        # prototypes per label
        q 2-z-run-supproto-direct-7ppl.sh "$ds" "$seed"
        q 2-z-run-supproto-direct-28ppl.sh "$ds" "$seed"
        q 3-z-run-protossl-heedb-pila-7ppl.sh "$ds" "$seed"
        q 3-z-run-protossl-heedb-pila-28ppl.sh "$ds" "$seed"
        q 4-z-run-supproto-heedb-rila-7ppl.sh "$ds" "$seed"
        q 4-z-run-supproto-heedb-rila-28ppl.sh "$ds" "$seed"

        # pretraining over HEEDB-150 labels, 7500 (150 labels x 50 ppl) and 1050 prototype banks
        q 3-run-protossl-heedb-150-pila.sh "$ds" "$seed"
        q 4-run-supproto-heedb-150-rila.sh "$ds" "$seed"
        q 3-run-protossl-heedb-150-1050-pila.sh "$ds" "$seed"
        q 4-run-supproto-heedb-150-1050-rila.sh "$ds" "$seed"

        # supproto no-proj
        q 4-y-run-supproto-heedb-pila.sh "$ds" "$seed"

        # assignment method
        q 6-run-protossl-heedb-pia.sh "$ds" "$seed"

        # no assignment
        q 3-z-run-protossl-heedb-pila-83ppl.sh "$ds" "$seed"
        q 6-z-run-protossl-heedb-pia-83ppl.sh "$ds" "$seed"
        q 7-1-run-protossl-heedb-pit.sh "$ds" "$seed"
        q 7-2-run-protossl-heedb-pip.sh "$ds" "$seed"
        q 7-3-run-protossl-heedb-latent.sh "$ds" "$seed"

        # start with pretrained encoder (each builds or reuses the node's ECGFounder patches)
        q 9-1-run-ecgfounder-lap.sh "$ds" "$seed"
        q 9-2-run-ecgfounder-clustering.sh "$ds" "$seed"
        q 9-3-run-ecgfounder-random.sh "$ds" "$seed"
      fi
    done
  done
done
