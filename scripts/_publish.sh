# Sourced by _common.sh and pretrain/_common.sh, which set RUN_DIR (node-local stage)
# and DEST_DIR (shared storage) first.

# stage <exp>...: check before training (rather than at publish time, hours later) that
# no exp was published yet, and clear any staged leftovers of an earlier attempt
stage() {
  local exp
  for exp in "$@"; do
    [[ ! -e $DEST_DIR/$exp ]] || { echo "$DEST_DIR/$exp already exists" >&2; exit 1; }
  done
  mkdir -p "$RUN_DIR"
  for exp in "$@"; do
    rm -rf "${RUN_DIR:?}/$exp"
  done
}

# publish <exp>: copy a finished experiment to shared storage, without wandb (it syncs to
# the wandb server, and it's many small files on NFS). Copied to a temp dir beside the
# target and renamed, so a partial copy is never visible under the final name.
publish() {
  local src=$RUN_DIR/$1 dest=$DEST_DIR/$1
  local tmp=$DEST_DIR/.$1.tmp-${SLURM_JOB_ID:-$$}
  mkdir -p "$tmp"
  tar -C "$src" --exclude=wandb -cf - . | tar -C "$tmp" -xf -
  echo "${PROTOSSL_COMMIT:-unknown}" >"$tmp/COMMIT"
  mv -T "$tmp" "$dest"
  echo "published $src to $dest"
}

# verify_sha256 <file> <sha256>
verify_sha256() {
  echo "$2  $1" | sha256sum --check --quiet ||
    { echo "$1 does not match sha256 $2" >&2; exit 1; }
}
