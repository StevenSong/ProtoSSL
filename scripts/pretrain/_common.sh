# Sourced by the pretraining scripts with their argument: <run>. Each pretrains on HEEDB,
# stages in $RUN_DIR (node-local) and publishes to $DEST_DIR = $SHARED_ROOT/runs/<run>,
# where the downstream experiments read it (see ../_common.sh).

(($# == 1)) || { echo "usage: $0 <run>" >&2; exit 1; }
RUN=$1

PRETRAIN_DATASET=/opt/gpudata/ecg/heedb

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${PROTOSSL_RUN_ROOT:-$HOME/protossl-run}
SHARED_ROOT=${PROTOSSL_SHARED_ROOT:-/shared/gpudata/songs1/protossl-shared}
RUN_DIR=$RUN_ROOT/outputs/$RUN
DEST_DIR=$SHARED_ROOT/runs/$RUN
source "$REPO_ROOT/scripts/_publish.sh"

echo "pretraining on $PRETRAIN_DATASET, run $RUN"
echo "staging in $RUN_DIR, publishing to $DEST_DIR"

cd "$REPO_ROOT/scripts"
