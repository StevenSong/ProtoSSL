"""Audit HEEDB physician-overread labels for statements that don't describe the current ECG.

The `original_physician` label source marks a label positive whenever one of its 12SL codes
appears anywhere in `codes_physician`. Many of those codes don't describe the current ECG. This
script tags every occurrence of every HEEDB_TARGETS code, then reduces it to one bucket per
(ECG, label): main, hedged, cmp_current, prior, cmp_other or dump. The rules live in
protossl/datasets/_heedb_label_rules.py, which the `audited_physician` label source also uses
(main + hedged + cmp_current, the "lenient" column below).

If an ECG has several occurrences, the lowest bucket wins, so "main" means at least one
occurrence was in the main read.

Outputs (to --out-dir):
    tags.parquet                  per-ECG bucket (0 = negative) for every label, plus metadata
    summary_counts.csv / _pct.csv per-label bucket counts and percentages of positives
    summary_by_split_source.csv   bucket counts per label x split x source
    review_samples.csv            random reads per (label, bucket) with physician and software
                                  text, label codes in **CAPS**, and a blank verdict column

Usage:
    CACHE_DIR=... python heedb_label_audit.py --out-dir DIR [--workers 128] [--n-review 20]
"""

import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from protossl.datasets import _heedb_dataset as heedb
from protossl.datasets._heedb_label_rules import (
    BUCKETS,
    WITH_CODES,
    make_code_to_labels,
    tag_read,
)
from protossl.defines import HEEDB_TARGETS

DATASET_PATH = "/opt/gpudata/ecg/heedb"
DICT_PATH = os.path.join(DATASET_PATH, "I0001/12SL_diagnoses/diagnoses_dictionary.csv")

_dict = pd.read_csv(DICT_PATH)
CODE_TEXT = dict(zip(_dict["codes"], _dict["diagnoses"].fillna("")))
assert WITH_CODES == {c for c, t in CODE_TEXT.items() if t.startswith("with ")}

LABELS = list(HEEDB_TARGETS)
CODE_TO_LABELS = make_code_to_labels(HEEDB_TARGETS)


def tag_chunk(code_strs) -> tuple[np.ndarray, list[str]]:
    tags = np.zeros((len(code_strs), len(LABELS)), dtype=np.uint8)
    kinds = []
    for i, s in enumerate(code_strs):
        tags[i], kd = tag_read(s, CODE_TO_LABELS, len(LABELS))
        kinds.append(kd)
    return tags, kinds


def render(code_str, highlight: set[int]) -> str:
    if not isinstance(code_str, str) or not code_str:
        return ""
    parts = []
    for c in map(int, code_str.split(",")):
        t = CODE_TEXT.get(c, f"<{c}>")
        parts.append(t.upper() if c in highlight else t)
    return " / ".join(parts)


def lookup_codes(meta: pd.DataFrame, label_src: heedb.LABEL_SRC_T) -> list:
    return heedb.lookup_codes(meta, *heedb.get_fname_to_code(DATASET_PATH, label_src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--n-review", type=int, default=20)
    ap.add_argument("--heedb-split-type", default="by-label")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta = heedb.get_heedb_metadata(
        DATASET_PATH, heedb_split_type=args.heedb_split_type
    )
    phys = lookup_codes(meta, "original_physician")
    soft = lookup_codes(meta, "original_muse")

    chunks = np.array_split(np.array(phys, dtype=object), args.workers * 8)
    with mp.Pool(args.workers) as pool:
        parts = pool.map(tag_chunk, chunks)
    tags = np.concatenate([t for t, _ in parts])
    kinds = [k for _, ks in parts for k in ks]

    df = meta[["ecg_id", "patient_id", "source", "year", "split", "fpath"]].copy()
    df["expansions"] = kinds
    for i, label in enumerate(LABELS):
        df[label] = tags[:, i]
    df.to_parquet(os.path.join(args.out_dir, "tags.parquet"), index=False)

    # per-label summary
    rows = []
    for label in LABELS:
        v = df[label]
        row = {"label": label, "positives": int((v > 0).sum())}
        row |= {name: int((v == b).sum()) for b, name in BUCKETS.items()}
        rows.append(row)
    counts = pd.DataFrame(rows).set_index("label")
    counts["strict"] = counts["main"]
    counts["lenient"] = counts[["main", "hedged", "cmp_current"]].sum(axis=1)
    counts.to_csv(os.path.join(args.out_dir, "summary_counts.csv"))
    pct = counts.drop(columns="positives").div(counts["positives"], axis=0) * 100
    pct.insert(0, "positives", counts["positives"])
    pct.round(1).to_csv(os.path.join(args.out_dir, "summary_pct.csv"))
    print(pct.round(1).sort_values("main").to_string())

    long = df.melt(
        id_vars=["split", "source"], value_vars=LABELS, var_name="label", value_name="b"
    )
    long = long[long["b"] > 0]
    long["bucket"] = long["b"].map(BUCKETS)
    by = (
        long.groupby(["label", "split", "source", "bucket"])
        .size()
        .unstack(fill_value=0)
    )
    by.to_csv(os.path.join(args.out_dir, "summary_by_split_source.csv"))

    # review samples, both reads rendered with the label's codes in CAPS
    rng = np.random.default_rng(0)
    review = []
    for label in LABELS:
        highlight = set(HEEDB_TARGETS[label])
        for b, name in BUCKETS.items():
            idx = np.flatnonzero(df[label].to_numpy() == b)
            if len(idx) == 0:
                continue
            for i in rng.choice(idx, min(args.n_review, len(idx)), replace=False):
                review.append(
                    {
                        "label": label,
                        "bucket": name,
                        "bucket_size": len(idx),
                        "ecg_id": df["ecg_id"].iat[i],
                        "source": df["source"].iat[i],
                        "year": df["year"].iat[i],
                        "split": df["split"].iat[i],
                        "fpath": df["fpath"].iat[i],
                        "physician_read": render(phys[i], highlight),
                        "software_read": render(soft[i], highlight),
                        "verdict": "",
                        "notes": "",
                    }
                )
    pd.DataFrame(review).to_csv(
        os.path.join(args.out_dir, "review_samples.csv"), index=False
    )
    print(f"wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
