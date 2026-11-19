# Sourced by _common.sh and 0-run-cache-data.sh. resolve_dataset <dataset> sets
# DATASET_PATH, e.g. ptbxl -> /opt/gpudata/ecg/ptb-xl, ptbxl-256 -> /opt/gpudata/ecg/ptb-xl-256.

# short name -> directory under /opt/gpudata/ecg; subsets append their size, e.g. -256
declare -A ECG_DATASET_DIRS=(
  [echonext]=echonext
  [ptbxl]=ptb-xl
  [cinc]=cinc-2020
  [mimic]=mimic-iv-ecg
  [zzu]=zzu-pecg
  [code15]=code15
)

resolve_dataset() {
  local base=${1%%-*}
  [[ -n ${ECG_DATASET_DIRS[$base]:-} ]] || { echo "unknown dataset: $1" >&2; exit 1; }
  DATASET_PATH=/opt/gpudata/ecg/${ECG_DATASET_DIRS[$base]}${1#"$base"}
  [[ -d $DATASET_PATH ]] || { echo "no dataset at $DATASET_PATH" >&2; exit 1; }
}
