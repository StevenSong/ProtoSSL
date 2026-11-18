# Sourced by the experiment scripts with their arguments: <dataset> <run> <seed>. Sets the
# paths the scripts use and provides the stage/publish steps.
#
#   <dataset>: ECG dataset short name, optionally with a train-size suffix, e.g. ptbxl or
#              ptbxl-256 (/opt/gpudata/ecg/ptb-xl-256)
#   <run>:     run name. Experiments are staged in $RUN_DIR (node-local) and published
#              to $DEST_DIR (shared storage), both laid out like the paper's archive:
#              <run>/protossl-outputs-seed<seed>/runs-<dataset>/<exp>
#   <seed>:    random seed
#
# Pretrained models are read from $PRETRAIN_DIR = $SHARED_ROOT/runs/<run>, where
# scripts/pretrain/ publishes them; set PROTOSSL_PRETRAIN_RUN to use another run's.

usage="usage: $0 <dataset> <run> <seed>"
(($# == 3)) || { echo "$usage" >&2; exit 1; }
DATASET=$1
RUN=$2
SEED=$3
[[ $SEED =~ ^[0-9]+$ ]] || { echo "seed must be an integer: $SEED" >&2; exit 1; }

source "$(dirname "${BASH_SOURCE[0]}")/_datasets.sh"
resolve_dataset "$DATASET"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${PROTOSSL_RUN_ROOT:-$HOME/protossl-run}
SHARED_ROOT=${PROTOSSL_SHARED_ROOT:-/opt/gpu_working/steven/protossl-shared}
seed_dir=$RUN/protossl-outputs-seed$SEED/runs-$DATASET
RUN_DIR=$RUN_ROOT/outputs/$seed_dir
DEST_DIR=$SHARED_ROOT/runs/$seed_dir
PRETRAIN_DIR=$SHARED_ROOT/runs/${PROTOSSL_PRETRAIN_RUN:-$RUN}

# foundation-model weights, pre-pulled on every node; jobs never download
ECGFOUNDER_CKPT=/opt/gpudata/models/PKUDigitalHealth/ECGFounder/12_lead_ECGFounder.pth
ECGFOUNDER_SHA256=ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997
STMEM_CKPT=/opt/gpudata/models/st-mem/st_mem_vit_base_encoder.pth
STMEM_SHA256=6f63d370fd1a2ccf2fbddd681571adf985eeae3f652973f0071342a39b7793d6

echo "dataset $DATASET ($DATASET_PATH), run $RUN, seed $SEED"
echo "staging in $RUN_DIR, publishing to $DEST_DIR"

source "$REPO_ROOT/scripts/_publish.sh"
cd "$REPO_ROOT/scripts"
