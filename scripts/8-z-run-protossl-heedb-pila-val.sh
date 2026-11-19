#!/usr/bin/env bash
#   slurm/submit.sh --gres=gpu:1 -w kg35-nvl01 -- scripts/8-z-run-protossl-heedb-pila-val.sh <dataset> <run> <seed>
# see scripts/_common.sh for the arguments and where outputs go
#
# Val-split counterpart of the inputs results/proto-recur-nonclassical.ipynb reads for
# train/test, for an independent replication (PILA never uses val: assignment,
# projection and the linear probe all fit on train):
#   protossl-heedb-pila-val:            <dataset>'s own PILA prototype similarities
#   protossl-heedb-pila-from-ptbxl-val: PTB-XL's statements imputed on <dataset>, as in
#                                       8-run-protossl-heedb-pila-from-ptbxl.sh
#
# Like pretrained models, the finished PILA experiments (<dataset>'s and PTB-XL's, at the
# same seed) are read from $PRETRAIN_DIR/protossl-outputs-seed<seed>/runs-<dataset>/.
set -euo pipefail
source "$(dirname "$0")/_common.sh" "$@"

[[ ${DATASET%%-*} != ptbxl ]] || { echo "$DATASET is PTB-XL, nothing to transfer" >&2; exit 1; }

# experiment parameters
OWN_EXP_NAME="protossl-heedb-pila-val"
PTBXL_EXP_NAME="protossl-heedb-pila-from-ptbxl-val"
SOURCE_DATASET_PATH=/opt/gpudata/ecg/ptb-xl
SOURCES=$PRETRAIN_DIR/protossl-outputs-seed$SEED
OWN_EXP=$SOURCES/runs-$DATASET/protossl-heedb-pila
PTBXL_EXP=$SOURCES/runs-ptbxl/protossl-heedb-pila
for exp in "$OWN_EXP" "$PTBXL_EXP"; do
  [[ -e $exp/project-prototypes-supervised/latest/proj.ckpt && -e $exp/model.joblib ]] ||
    { echo "no finished PILA experiment at $exp" >&2; exit 1; }
done

# compute_val_embeddings <exp-name> <source-exp>: val similarities to the source's
# projected prototypes (one projection row per prototype)
compute_val_embeddings() {
  local proj=$2/project-prototypes-supervised/latest
  python -m protossl.trainer \
      --config $REPO_ROOT/configs/target-guided-14ppl.yaml \
      --seed_everything $SEED \
      --pipeline-stage compute-embeddings \
      --embedding-splits '[val]' \
      --model.n_prototypes_per_label null \
      --model.n_prototypes $(($(wc -l <"$proj/projection_metadata.csv") - 1)) \
      --trainer.logger.save_dir $RUN_DIR \
      --trainer.logger.name $1 \
      --data.dataset_path $DATASET_PATH \
      --model.pretrained_weights $proj/proj.ckpt
  echo "$2" >"$RUN_DIR/$1/SOURCE"
}

stage "$OWN_EXP_NAME" "$PTBXL_EXP_NAME"

compute_val_embeddings "$OWN_EXP_NAME" "$OWN_EXP"

compute_val_embeddings "$PTBXL_EXP_NAME" "$PTBXL_EXP"
python _transfer_linear_probe.py \
--source-experiment $PTBXL_EXP \
--source-dataset-path $SOURCE_DATASET_PATH \
--embeddings $RUN_DIR/$PTBXL_EXP_NAME/compute-embeddings/latest \
--output-path $RUN_DIR/$PTBXL_EXP_NAME \
--splits val

# publish both at the end, so a failed job leaves nothing published
publish "$OWN_EXP_NAME"
publish "$PTBXL_EXP_NAME"
