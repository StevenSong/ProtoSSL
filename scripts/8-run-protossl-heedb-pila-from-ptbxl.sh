#!/usr/bin/env bash
#   slurm/submit.sh --gres=gpu:1 -- scripts/8-run-protossl-heedb-pila-from-ptbxl.sh <dataset> <run> <seed>
# see scripts/_common.sh for the arguments and where outputs go
#
# Reads <dataset>'s ECGs through PTB-XL's PILA model at the same seed: similarities to
# PTB-XL's projected prototypes, then PTB-XL's linear probe, which imputes PTB-XL's
# statements on <dataset>. They stand in for machine-read statements, as known ECG
# findings, in datasets that ship without them (results/proto-recur-nonclassical.ipynb).
#
# Like a pretrained model, the PTB-XL experiment is read from $PRETRAIN_DIR, at
# <pretrain-run>/protossl-outputs-seed<seed>/runs-ptbxl/protossl-heedb-pila.
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

[[ ${DATASET%%-*} != ptbxl ]] || { echo "$DATASET is PTB-XL, nothing to transfer" >&2; exit 1; }

# experiment parameters
EXP_NAME="protossl-heedb-pila-from-ptbxl"
SOURCE_DATASET_PATH=/opt/gpudata/ecg/ptb-xl
SOURCE_EXP=$PRETRAIN_DIR/protossl-outputs-seed$SEED/runs-ptbxl/protossl-heedb-pila
SOURCE_PROJ=$SOURCE_EXP/project-prototypes-supervised/latest
[[ -e $SOURCE_PROJ/proj.ckpt && -e $SOURCE_EXP/model.joblib ]] ||
  { echo "no finished PTB-XL PILA experiment at $SOURCE_EXP" >&2; exit 1; }
# one projection row per prototype; the target dataset's label count doesn't apply
N_PROTOTYPES=$(($(wc -l <"$SOURCE_PROJ/projection_metadata.csv") - 1))

stage "$EXP_NAME"

python -m protossl.trainer \
    --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
    --seed_everything $SEED \
    --pipeline-stage compute-embeddings \
    --model.n_prototypes_per_label null \
    --model.n_prototypes $N_PROTOTYPES \
    --trainer.logger.save_dir $RUN_DIR \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_path $DATASET_PATH \
    --model.pretrained_weights $SOURCE_PROJ/proj.ckpt

python _transfer_linear_probe.py \
--source-experiment $SOURCE_EXP \
--source-dataset-path $SOURCE_DATASET_PATH \
--embeddings $RUN_DIR/$EXP_NAME/compute-embeddings/latest \
--output-path $RUN_DIR/$EXP_NAME

echo "$SOURCE_EXP" >"$RUN_DIR/$EXP_NAME/SOURCE"

publish "$EXP_NAME"
