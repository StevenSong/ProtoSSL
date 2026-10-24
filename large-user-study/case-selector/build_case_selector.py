"""Build the offline HEEDB case selector bundle for picking user study cases.

Writes a self-contained folder that opens by double-clicking `index.html`, no
install or server needed:

    OUT_DIR/
      index.html               static viewer (copied from this directory)
      index.js                 ECG IDs, metadata, and label bitmasks for every label source
                               (physician overread, original MUSE, ProtoECGNet fusion predictions)
      HOW_TO_USE.txt           instructions for the person selecting cases
      pdf/<ecg_id // 10000>/<ecg_id>.pdf   clinical printout from `plot_ecg`

Rendering is resumable, PDFs already on disk are skipped, and `index.js` is
written last so it can flag any ECG whose PDF is missing.

Usage:
    CACHE_DIR=... python build_case_selector.py --out-dir DIR [--workers 64] [--limit N]
        [--years 2018-2022] [--fusion-dir RUN_DIR/fusion/train-fusion-classifier/latest]
"""

import os

# one thread per worker, parallelism comes from the process pool
for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

import argparse
import datetime
import json
import multiprocessing as mp
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve
from tqdm import tqdm

from protossl.datasets import HeedbECGDataset, reject_unknown_labels
from protossl.datasets._heedb_dataset import get_heedb_labels
from protossl.defines import HEEDB_TARGETS
from protossl.plotting import plot_ecg

DATASET_PATH = "/opt/gpudata/ecg/heedb"
# same split as the user study models
DATASET_KWARGS = dict(
    dataset_path=DATASET_PATH,
    split="test",
    sampling_rate=500,
    heedb_split_type="by-label",
    drop_blank_overread=True,
)
# first source is the default filter in the viewer
LABEL_SOURCES = [
    ("original_physician", "Physician overread", "Phys."),
    ("original_muse", "MUSE (original)", "MUSE"),
    ("protoecgnet_fusion", "ProtoECGNet fusion (model)", "Fusion"),
]
# patient-level split retrain, see large-user-study/scripts/run-protoecgnet-heedb-fusion.sh
FUSION_DIR = (
    "/opt/gpu_working/steven/protoecgnet-heedb-v3/fusion/train-fusion-classifier/latest"
)
SHARD_SIZE = 10_000
HERE = Path(__file__).resolve().parent

# set in the parent before forking so workers share the loaded dataset
_DS: HeedbECGDataset | None = None


def parse_years(spec: str) -> list[int]:
    """Parse a year spec like "2018-2022" or "2013,2016-2018" into sorted years, inclusive."""
    years = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            lo_y, hi_y = int(lo), int(hi if sep else lo)
        except ValueError:
            raise argparse.ArgumentTypeError(f"bad year or year range: {part!r}")
        if lo_y > hi_y:
            raise argparse.ArgumentTypeError(f"year range is reversed: {part!r}")
        years.update(range(lo_y, hi_y + 1))
    if not years:
        raise argparse.ArgumentTypeError(f"no years in {spec!r}")
    return sorted(years)


def fusion_predictions(fusion_dir: str) -> tuple[np.ndarray, list[float]]:
    """Binarize fusion test probabilities at per-label Youden's J thresholds from val.

    Same thresholding as large-user-study/notebooks/muse-accuracy.ipynb.
    """
    assert _DS is not None
    n_labels = len(HEEDB_TARGETS)
    test_probs = np.load(os.path.join(fusion_dir, "test_probs.npy"))
    val_probs = np.load(os.path.join(fusion_dir, "val_probs.npy"))
    assert test_probs.shape == (len(_DS), n_labels), (
        f"test_probs.npy is {test_probs.shape}, expected {(len(_DS), n_labels)}, "
        "was the model trained on a different split?"
    )
    ds_val = HeedbECGDataset(**{**DATASET_KWARGS, "split": "val"})  # type: ignore
    y_val = ds_val.labels.numpy()
    reject_unknown_labels(y_val, "case selector thresholds")
    assert val_probs.shape == y_val.shape, (val_probs.shape, y_val.shape)

    thresholds = []
    for i in range(n_labels):
        fpr, tpr, thr = roc_curve(y_val[:, i], val_probs[:, i])
        thresholds.append(float(thr[np.argmax(tpr - fpr)]))
    return (test_probs > np.asarray(thresholds)).astype(np.int64), thresholds


def load_label_sources(fusion_dir: str) -> tuple[dict[str, np.ndarray], dict]:
    """(N, L) 0/1 labels for every label source over the full split, plus fusion provenance."""
    assert _DS is not None
    sources, fusion_meta = {}, {}
    for key, _, _ in LABEL_SOURCES:
        if key == "original_physician":
            y = _DS.labels.numpy()
        elif key == "protoecgnet_fusion":
            y, thresholds = fusion_predictions(fusion_dir)
            fusion_meta = {
                "dir": fusion_dir,
                "thresholds": dict(zip(HEEDB_TARGETS, thresholds)),
            }
        else:
            y = get_heedb_labels(DATASET_PATH, _DS._df, None, key)  # type: ignore
        assert y.shape == (len(_DS), len(HEEDB_TARGETS)), (key, y.shape)
        # the viewer stores labels as 0/1 bitmasks
        reject_unknown_labels(y, f"case selector ({key})")
        sources[key] = y
    return sources, fusion_meta


def pdf_relpath(ecg_id: int) -> str:
    # keep in sync with pdfPath() in index.html
    return f"pdf/{ecg_id // SHARD_SIZE:03d}/{ecg_id}.pdf"


def _init_worker():
    torch.set_num_threads(1)


def _render_chunk(args: tuple[str, list[int]]) -> tuple[int, list[tuple[int, str]]]:
    out_dir, rows = args
    assert _DS is not None
    n_done, failures = 0, []
    for i in rows:
        ecg_id = int(_DS.sample_ids[i])
        path = os.path.join(out_dir, pdf_relpath(ecg_id))
        try:
            fig = plot_ecg(dataset=_DS, index=i)
            # write then rename so an interrupted run never leaves a truncated PDF
            tmp = f"{path}.tmp"
            fig.savefig(tmp, format="pdf")
            os.replace(tmp, path)
            n_done += 1
        except Exception as e:
            failures.append((ecg_id, repr(e)))
        finally:
            plt.close("all")
    return n_done, failures


def render(out_dir: str, rows: np.ndarray, workers: int, chunk_size: int = 64):
    assert _DS is not None
    ecg_ids = _DS.sample_ids.numpy()
    for shard in np.unique(ecg_ids[rows] // SHARD_SIZE):
        os.makedirs(os.path.join(out_dir, f"pdf/{shard:03d}"), exist_ok=True)
    todo = [
        int(i)
        for i in rows
        if not os.path.exists(os.path.join(out_dir, pdf_relpath(int(ecg_ids[i]))))
    ]
    print(f"{len(rows) - len(todo)} PDFs already on disk, rendering {len(todo)}")
    if not todo:
        return

    chunks = [
        (out_dir, todo[s : s + chunk_size]) for s in range(0, len(todo), chunk_size)
    ]
    start = time.time()
    n_done, failures = 0, []
    with mp.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
        with tqdm(total=len(todo), unit="ecg") as pbar:
            for done, failed in pool.imap_unordered(_render_chunk, chunks):
                n_done += done
                failures.extend(failed)
                pbar.update(done + len(failed))
    elapsed = time.time() - start
    print(
        f"rendered {n_done} PDFs in {elapsed:.0f}s "
        f"({n_done / elapsed:.1f} ECG/s with {workers} workers)"
    )
    for ecg_id, err in failures:
        print(f"FAILED ecg_id={ecg_id}: {err}")


def write_index(
    out_dir: str,
    rows: np.ndarray,
    sources: dict[str, np.ndarray],
    fusion_meta: dict,
    years_spec: str | None = None,
):
    assert _DS is not None
    df = _DS._df.iloc[rows]
    ecg_ids = df["ecg_id"].to_numpy()
    labels = list(HEEDB_TARGETS)
    bits = np.int64(1) << np.arange(len(labels), dtype=np.int64)

    masks = {
        key: (y[rows].astype(np.int64) * bits).sum(1).tolist()
        for key, y in sources.items()
    }

    missing = [
        int(e)
        for e in ecg_ids
        if not os.path.exists(os.path.join(out_dir, pdf_relpath(int(e))))
    ]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    payload = {
        "meta": {
            "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "commit": commit,
            "dataset": {k: v for k, v in DATASET_KWARGS.items() if k != "dataset_path"},
            "years": years_spec,
            "fusion": fusion_meta,
            "n_ecgs": len(rows),
            "n_total_in_split": len(_DS),
        },
        "labels": labels,
        "label_sources": [
            {"key": k, "name": n, "short": s} for k, n, s in LABEL_SOURCES
        ],
        "ecg_id": ecg_ids.tolist(),
        "patient_id": df["patient_id"].tolist(),
        "age": np.nan_to_num(df["age"].to_numpy(), nan=-1).round().astype(int).tolist(),
        "year": df["year"].tolist(),
        # one character per ECG, M/F/U and m(gb)/e(mory)
        "sex": "".join(
            s[0] if s[:1] in ("M", "F") else "U"
            for s in df["sex"].fillna("U").astype(str).str.upper()
        ),
        "site": "".join(df["source"].str[0]),
        "masks": masks,
        "missing": missing,
    }
    path = os.path.join(out_dir, "index.js")
    with open(f"{path}.tmp", "w") as f:
        f.write("window.HEEDB_INDEX = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    os.replace(f"{path}.tmp", path)
    size_mb = os.path.getsize(path) / 2**20
    print(
        f"wrote {path} ({len(rows)} ECGs, {len(missing)} missing PDFs, {size_mb:.1f} MB)"
    )


def main():
    global _DS
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only build a random subset of this many ECGs, for timing and testing",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="only include ECGs acquired in these years, inclusive, e.g. 2018-2022 or "
        "2013,2016-2018. Applied before --limit",
    )
    parser.add_argument(
        "--fusion-dir",
        default=FUSION_DIR,
        help="train-fusion-classifier log dir with test_probs.npy and val_probs.npy",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="skip rendering, only rewrite index.js",
    )
    args = parser.parse_args()
    try:
        years = parse_years(args.years) if args.years is not None else None
    except argparse.ArgumentTypeError as e:
        parser.error(f"--years: {e}")

    for fname in ["test_probs.npy", "val_probs.npy"]:
        if not os.path.exists(os.path.join(args.fusion_dir, fname)):
            parser.error(f"--fusion-dir {args.fusion_dir} has no {fname}")

    _DS = HeedbECGDataset(**DATASET_KWARGS)  # type: ignore
    # before rendering so a bad --fusion-dir fails fast
    sources, fusion_meta = load_label_sources(args.fusion_dir)
    rows = np.arange(len(_DS))
    if years is not None:
        ds_years = _DS._df["year"].to_numpy()
        absent = sorted(set(years) - set(ds_years.tolist()))
        if absent:
            print(f"WARNING: no ECGs in the split for years {absent}")
        rows = rows[np.isin(ds_years, years)]
        counts = dict(zip(*np.unique(ds_years[rows], return_counts=True)))
        print(
            f"year filter {args.years}: {len(rows)} of {len(_DS)} ECGs, "
            + ", ".join(f"{y}: {n}" for y, n in counts.items())
        )
        if len(rows) == 0:
            parser.error(f"--years {args.years} matches no ECGs")
    if args.limit is not None:
        rng = np.random.default_rng(args.seed)
        rows = np.sort(rng.choice(rows, size=min(args.limit, len(rows)), replace=False))

    os.makedirs(args.out_dir, exist_ok=True)
    for fname in ["index.html", "HOW_TO_USE.txt"]:
        shutil.copy2(HERE / fname, os.path.join(args.out_dir, fname))
    if not args.index_only:
        render(args.out_dir, rows, args.workers)
    write_index(args.out_dir, rows, sources, fusion_meta, args.years)


if __name__ == "__main__":
    main()
