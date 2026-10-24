import ast
import hashlib
import inspect
import json
import os
import textwrap
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..defines import (
    CACHE_DIR,
    HEEDB_EUH_CLIPPED_MEANS,
    HEEDB_EUH_CLIPPED_STDS,
    HEEDB_EUH_LOWERS,
    HEEDB_EUH_UPPERS,
    HEEDB_LEAD_ORDER,
    HEEDB_MGB_CLIPPED_MEANS,
    HEEDB_MGB_CLIPPED_STDS,
    HEEDB_MGB_LOWERS,
    HEEDB_MGB_UPPERS,
    HEEDB_TARGETS,
    SPLIT_T,
    STANDARD_LEAD_ORDER,
    USE_HEEDB_150,
    env_flag,
)
from ._base_dataset import (
    BaseTSDataset,
    atomic_write,
    get_cache_file,
    load_cached_data,
    validate_label_subset,
)
from ._heedb_label_rules import (
    CURRENT_BUCKETS,
    RULES_HASH,
    VALIDATED_TARGETS,
    make_code_to_labels,
    tag_read,
)
from .streaming_loaders import StreamingECGWaveforms, _streaming_ecgs

# fmt: off
LABEL_SRC_MAPPING = {
    "original_muse":      ("diagnoses_acquisition.csv", "codes_software"),
    "original_physician": ("diagnoses_acquisition.csv", "codes_physician"),
    "new_muse":           ("diagnoses_v24.csv",         "codes"),
    # physician codes that may describe the current ECG, see _heedb_label_rules.py
    "audited_physician":  ("diagnoses_acquisition.csv", "codes_physician"),
}
# fmt: on
LABEL_SRC_T = Literal[
    "original_muse", "original_physician", "new_muse", "audited_physician"
]
HEEDB_SPLIT_T = Literal["by-year", "by-label"]
BY_LABEL_SPLIT_HASH = "543697e39036d4bc2ee97e14b86ae24c7c7cf83def07fe70206ae6dcafd4ed56"  # pragma: allowlist secret

heedb_lead_order = [l.lower() for l in HEEDB_LEAD_ORDER]
standard_lead_order = [l.lower() for l in STANDARD_LEAD_ORDER]
assert all([c == s for c, s in zip(heedb_lead_order, standard_lead_order)])

# per-institution normalization, keyed by the `source` column
STREAM_STATS = {
    "per_lead_lowerbound": {"mgb": HEEDB_MGB_LOWERS, "emory": HEEDB_EUH_LOWERS},
    "per_lead_upperbound": {"mgb": HEEDB_MGB_UPPERS, "emory": HEEDB_EUH_UPPERS},
    "per_lead_mean": {"mgb": HEEDB_MGB_CLIPPED_MEANS, "emory": HEEDB_EUH_CLIPPED_MEANS},
    "per_lead_std": {"mgb": HEEDB_MGB_CLIPPED_STDS, "emory": HEEDB_EUH_CLIPPED_STDS},
}

# DANGER: set this to load the entire waveform database into memory
# DANGER: the initial cache should be done prior to any jobs
HIGH_MEMORY = env_flag("HIGH_MEMORY")

# global cache variables
FULL_META: dict[tuple[str, str], pd.DataFrame] = dict()  # (path, split type) -> meta
# (path, institution, csv, col) -> fname -> code string
FNAME_TO_CODE: dict[tuple[str, str, str, str], dict[str, str]] = dict()


class HeedbECGDataset(BaseTSDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
        label_src: LABEL_SRC_T = "original_physician",
        heedb_split_type: HEEDB_SPLIT_T = "by-year",
        drop_blank_overread: bool = False,
    ):
        full_df = get_heedb_metadata(dataset_path, heedb_split_type=heedb_split_type)

        split_df = full_df[full_df["split"] == split].reset_index(drop=True)
        df = split_df
        has_overread = None
        if drop_blank_overread:
            # HEEDB v5 likely has some errors with missing physician overreads,
            # so a blank physician overread can't be told apart from a physician
            # deleting every statement. Thus, treat these rows as unlabeled
            # rather than all-negative. filtered after split assignment so splits
            # are unchanged and ecg_id (sample_id) stays stable. the mask is always
            # derived from physician overreads, so the same subset of ECGs can be
            # labeled with any label_src (e.g. muse labels over the overread subset)
            has_overread = get_heedb_overread_mask(dataset_path, split_df)
            df = split_df[has_overread].reset_index(drop=True)
        self.source_ids = torch.as_tensor(df["patient_id"].to_numpy())
        self.sample_ids = torch.as_tensor(df["ecg_id"].to_numpy())
        self.labels = torch.as_tensor(
            get_heedb_labels(dataset_path, df, label_subset, label_src)
        )
        self._df = df

        streaming_ecgs = StreamingECGWaveforms(
            wfdb_paths=list(df["full_path"]),
            sampling_rate=sampling_rate,
            **STREAM_STATS,
            verbose=not HIGH_MEMORY,
            stat_mapper=list(df["source"]),
            expected_lead_order=heedb_lead_order,
        )

        if not HIGH_MEMORY:
            self.waveforms = streaming_ecgs
        else:
            print("==================HeedbECGDataset==================")
            print("WARNING:")
            print("WARNING: ABOUT TO LOAD ENTIRE HEEDB WAVEFORM MATRIX INTO MEMORY")
            print("WARNING: THIS WILL CONSUME APPROXIMATELY 500 GB OF RAM IN THE JOB")
            print("WARNING:")
            print("===================================================")

            base_split = (
                split
                if heedb_split_type == "by-year"
                else f"{split}-{heedb_split_type}"
            )
            unfiltered_cache = get_cache_file(
                dataset_path=dataset_path,
                split=base_split,  # type: ignore
                sampling_rate=sampling_rate,
                key_extra=waveform_cache_key(split_df),
            )
            cache_split = (
                base_split
                if not drop_blank_overread
                else f"{base_split}-drop-blank-overread"
            )

            def load_transform_data_fn() -> torch.Tensor:
                if drop_blank_overread and os.path.exists(unfiltered_cache):
                    # rows of the unfiltered cache are the unfiltered split df in order, so
                    # index it with the overread mask instead of re-streaming every file.
                    # mmap keeps peak memory at roughly the size of the filtered tensor
                    print(f"deriving filtered cache from: {unfiltered_cache}")
                    X = torch.load(unfiltered_cache, mmap=True)
                    assert has_overread is not None
                    assert X.shape[0] == len(has_overread), (
                        f"unfiltered cache has {X.shape[0]} rows, "
                        f"expected {len(has_overread)}"
                    )
                    return X[torch.from_numpy(has_overread)]
                # fmt: off
                print("WARNING:")
                print("WARNING: ABOUT TO LOAD ENTIRE HEEDB WAVEFORM MATRIX INTO MEMORY TO CACHE")
                print("WARNING: THIS USES A DATALOADER AND SHOULD NOT BE DONE INSIDE A TRAINING JOB")
                print("WARNING:")
                # fmt: on
                dl = DataLoader(
                    streaming_ecgs,  # type: ignore
                    batch_size=512,
                    num_workers=8,
                    prefetch_factor=4,
                )
                data = []
                for batch in tqdm(dl):
                    data.append(batch)
                X = torch.concatenate(data)
                return X

            self.waveforms = load_cached_data(
                load_transform_data_fn=load_transform_data_fn,
                dataset_path=dataset_path,
                split=cache_split,  # type: ignore
                sampling_rate=sampling_rate,
                key_extra=waveform_cache_key(df),
            )

        assert self.source_ids.shape[0] == self.waveforms.shape[0]
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]
        assert self.source_ids.shape[0] == self.labels.shape[0]


def get_heedb_metadata(
    heedb_path: str,
    *,  # enforce kwargs
    heedb_split_type: HEEDB_SPLIT_T = "by-year",
) -> pd.DataFrame:
    _path = Path(heedb_path)
    meta_key = (heedb_path.rstrip(os.sep), heedb_split_type)
    if meta_key in FULL_META:
        return FULL_META[meta_key].copy()

    print("================get_heedb_metadata=================")
    print(f"using {heedb_split_type} splits")

    # the function's own source, so editing filters or split logic rebuilds the cache.
    # downstream caches are keyed on rows, so they survive a rebuild that keeps the rows
    code_hash = source_code_hash(get_heedb_metadata)
    src_hash = source_fingerprint(
        [_path / "I0001/metadata/metadata.csv", _path / "I0006/metadata/metadata.csv"]
    )
    identifier = (
        f"HEEDB_{heedb_path.rstrip(os.sep)}_{heedb_split_type}_{code_hash}_{src_hash}"
    )
    hashed = hashlib.md5(identifier.encode("utf-8")).hexdigest()[:8]
    cache_file = os.path.join(CACHE_DIR, f"{hashed}.csv")
    if os.path.exists(cache_file):
        print(f"reading HEEDB metadata from on-disk cache: {cache_file}")
        # round_trip so ages match a fresh build exactly, the default parser can be 1 ulp off
        df = pd.read_csv(cache_file, float_precision="round_trip")
    else:
        print(f"reading HEEDB metadata from source: {heedb_path}")

        # read harvard data
        mgb = pd.read_csv(
            _path / "I0001/metadata/metadata.csv",
            usecols=["BDSPPatientID", "SexDSC", "AgeAtAcquisition", "FileName"],
        )
        mgb["AgeAtAcquisition"] = mgb["AgeAtAcquisition"] / 365.2425
        mgb = mgb[(mgb["AgeAtAcquisition"] >= 18) & (mgb["SexDSC"].notna())]
        mgb = mgb.rename(
            columns={
                "BDSPPatientID": "patient_id",
                "SexDSC": "sex",
                "AgeAtAcquisition": "age",
                "FileName": "fpath",
            }
        )
        print("read MGB data")

        # read emory data, slight differences
        emory = pd.read_csv(
            _path / "I0006/metadata/metadata.csv",
            usecols=["BDSPPatientID", "Sex", "AgeAtAcquisition", "FileName"],
        )
        emory["AgeAtAcquisition"] = emory["AgeAtAcquisition"] / 365.2425
        emory = emory[
            (emory["AgeAtAcquisition"] >= 18)
            & (emory["Sex"].notna())
            & (emory["BDSPPatientID"].notna())
        ]
        emory = emory.rename(
            columns={
                "BDSPPatientID": "patient_id",
                "Sex": "sex",
                "AgeAtAcquisition": "age",
                "FileName": "fpath",
            }
        )
        assert (emory["patient_id"].astype(int) == emory["patient_id"]).all()
        emory["patient_id"] = emory["patient_id"].astype(int)

        # bad files on emory side
        emory_exclude = {"WFDB/2013/MUSE_20200225_081000_06000"}
        emory = emory[~emory["fpath"].isin(emory_exclude)]
        print("read Emory data")

        # join together
        mgb["source"] = "mgb"
        emory["source"] = "emory"
        assert len(set(mgb["patient_id"]) & set(emory["patient_id"])) == 0
        df = pd.concat([mgb, emory], ignore_index=True)  # MGB, then EUH
        df.index.name = "ecg_id"
        df["year"] = df["fpath"].str[1:].str.split("/").str[1].astype(int)
        df = df.reset_index()[
            ["ecg_id", "patient_id", "age", "sex", "year", "source", "fpath"]
        ]

        df["split"] = "train"
        if heedb_split_type == "by-year":
            # emory data ends in 2018 so val/test are all MGB data
            df.loc[df["year"] == 2021, "split"] = "val"
            df.loc[df["year"] == 2022, "split"] = "test"
        elif heedb_split_type == "by-label":
            # law of large numbers, we just generate random splits and it's close enough
            # to splits which preserve independent label prevalence (need to check about cooccurrence).
            # split by patient so that no patient's ECGs span multiple splits
            patient_ids = np.sort(df["patient_id"].unique()).astype(np.int64)
            n = int(len(patient_ids) * 0.05)
            rng = np.random.default_rng(42)
            val_test_patients = rng.choice(patient_ids, size=n * 2, replace=False)
            digest = hashlib.sha256(val_test_patients.tobytes()).hexdigest()
            if digest != BY_LABEL_SPLIT_HASH:
                raise ValueError(
                    "Randomly generated split for preserving label prevalence has changed!"
                )

            df.loc[df["patient_id"].isin(val_test_patients[:n]), "split"] = "val"
            df.loc[df["patient_id"].isin(val_test_patients[n:]), "split"] = "test"
        else:
            raise ValueError(
                f"Unknown how to create splits for HEEDB split type: {heedb_split_type}"
            )

        full_paths = []
        for f, src in zip(df["fpath"], df["source"]):
            if src == "mgb":
                # harvard paths start with "/S...", the level just under WFDB
                p = _path / "I0001/WFDB" / f[1:]
            elif src == "emory":
                # emory paths start with "WFDB/..."
                p = _path / "I0006" / f
            else:
                raise ValueError(f"Unknown path structure for institution: {src}")
            # str, to match what reading the cached csv gives
            full_paths.append(str(p))
        df["full_path"] = full_paths
        os.makedirs(CACHE_DIR, exist_ok=True)
        atomic_write(cache_file, lambda f: df.to_csv(f, index=False))
        print(f"saved HEEDB metadata to on-disk cache: {cache_file}")

    FULL_META[meta_key] = df.copy()
    print("===================================================")
    return df


def get_heedb_labels(
    heedb_path: str,
    meta: pd.DataFrame,
    label_subset: list[str] | None = None,
    label_src: LABEL_SRC_T = "original_physician",
) -> np.ndarray:
    print("=================make_heedb_labels=================")
    targets = HEEDB_TARGETS
    if label_subset is not None:
        validate_label_subset(label_subset, list(HEEDB_TARGETS))
        targets = {label: HEEDB_TARGETS[label] for label in label_subset}

    if label_src == "audited_physician":
        validate_audited_targets(targets)

    meta_hash = hash_path_list(meta["fpath"])
    # names and codes, so editing a target's codes doesn't reuse a stale cache
    targets_hash = hashlib.md5(json.dumps(targets).encode("utf-8")).hexdigest()[:8]
    src_hash = label_src_fingerprint(heedb_path, label_src)
    identifier = f"HEEDB_labels_{heedb_path.rstrip(os.sep)}_{meta_hash}_{label_src}_{targets_hash}_{src_hash}"
    if label_src == "audited_physician":
        identifier += f"_{RULES_HASH}"
    hashed = hashlib.md5(identifier.encode("utf-8")).hexdigest()[:8]
    cache_file = os.path.join(CACHE_DIR, f"{hashed}.npy")

    if os.path.exists(cache_file):
        print(f"reading HEEDB labels from on-disk cache: {cache_file}")
        loaded = np.load(cache_file, allow_pickle=True).item()
        data = loaded["data"]
        n_ecgs, n_annots = loaded["n_ecgs"], loaded["n_annots"]
        n_matched = loaded["n_matched"]
        print("replaying below stats from cache:")
    else:
        print(f"reading HEEDB labels from source: {heedb_path}")
        mgb_codes, euh_codes = get_fname_to_code(heedb_path, label_src)
        code_to_labels = make_code_to_labels(targets)

        data = np.zeros((len(meta), len(targets)), dtype=np.long)
        count = 0
        for meta_idx, codes in enumerate(
            tqdm(
                lookup_codes(meta, mgb_codes, euh_codes),
                total=len(meta),
                desc="Converting code string to labels",
            )
        ):
            if codes is None:
                # separate missing vs empty (below)
                continue
            count += 1
            if isinstance(codes, float) and np.isnan(codes):
                continue
            assert isinstance(codes, str)
            if label_src == "audited_physician":
                buckets, _ = tag_read(codes, code_to_labels, len(targets))
                data[meta_idx] = np.isin(buckets, CURRENT_BUCKETS)
                continue
            for code in codes.split(","):
                # convert 12SL code to label column idxs
                for label_idx in code_to_labels.get(int(code), []):
                    data[meta_idx, label_idx] = 1

        n_ecgs, n_matched = len(meta), count
        n_annots = len(mgb_codes) + len(euh_codes)
        os.makedirs(CACHE_DIR, exist_ok=True)
        saved = {
            "data": data,
            "n_ecgs": n_ecgs,
            "n_annots": n_annots,
            "n_matched": n_matched,
        }
        atomic_write(cache_file, lambda f: np.save(f, saved))  # type: ignore
        print(f"saved HEEDB labels to on-disk cache: {cache_file}")
    print(
        f"Of {n_ecgs} ECGs and {n_annots} annotations, {n_matched} matched "
        f"({n_ecgs-n_matched} ECG annotations were missing and filled with 0s)"
    )
    print("===================================================")
    return data


def make_fname_to_code(
    heedb_path: str,
    institution: str,
    label_csv: str,
    label_col: str,
) -> dict[str, str]:
    if institution == "mgb":
        subdir = "I0001"
    elif institution == "emory":
        subdir = "I0006"
    else:
        raise ValueError(f"Unknown subdir for institution: {institution}")
    df = pd.read_csv(os.path.join(heedb_path, subdir, "12SL_diagnoses", label_csv))
    return {
        fname: code_str
        for fname, code_str in zip(
            tqdm(
                # MGB v24 file names are prefixed with '.' and suffixed with '.hea\n'
                df["FileName"].str.removeprefix(".").str.removesuffix(".hea\n"),
                desc=f"Creating {institution} fname to code mapping",
            ),
            df[label_col],
        )
    }


def get_fname_to_code(
    heedb_path: str, label_src: LABEL_SRC_T
) -> tuple[dict[str, str], dict[str, str]]:
    # (MGB, Emory) fname -> code string maps, cached in a global and shared across label
    # sources that read the same csv column
    label_csv, label_col = LABEL_SRC_MAPPING[label_src]
    maps = []
    for institution in ("mgb", "emory"):
        key = (heedb_path.rstrip(os.sep), institution, label_csv, label_col)
        if key not in FNAME_TO_CODE:
            FNAME_TO_CODE[key] = make_fname_to_code(
                heedb_path, institution, label_csv, label_col
            )
        maps.append(FNAME_TO_CODE[key])
    return maps[0], maps[1]


def lookup_codes(
    meta: pd.DataFrame, mgb_codes: dict[str, str], euh_codes: dict[str, str]
) -> list:
    # code string per meta row, None if the file isn't in the diagnoses table
    out = []
    for fname, institution in zip(meta["fpath"], meta["source"]):
        if institution == "mgb":
            out.append(mgb_codes.get(fname))
        elif institution == "emory":
            out.append(euh_codes.get(fname))
        else:
            raise ValueError(f"Unknown institution: {institution}")
    return out


def validate_audited_targets(targets: dict[str, list[int]]):
    # the tagging rules were only checked on the 20 HEEDB targets and their codes
    if USE_HEEDB_150:
        raise ValueError(
            "label_src='audited_physician' was only validated on the 20 HEEDB targets, "
            "unset USE_HEEDB_150 to use it"
        )
    for label, codes in targets.items():
        if VALIDATED_TARGETS.get(label) != codes:
            raise ValueError(
                f"label_src='audited_physician' was not validated for {label}: {codes}"
            )


def get_heedb_overread_mask(heedb_path: str, meta: pd.DataFrame) -> np.ndarray:
    """
    Boolean mask over `meta` rows, True where the ECG has a non-blank physician
    overread (`codes_physician` in diagnoses_acquisition.csv). Rows whose file isn't
    in the diagnoses table at all are also False, but are counted separately.
    """
    print("==============get_heedb_overread_mask==============")
    label_src: LABEL_SRC_T = "original_physician"
    meta_hash = hash_path_list(meta["fpath"])
    src_hash = label_src_fingerprint(heedb_path, label_src)
    identifier = f"HEEDB_overread_mask_{heedb_path.rstrip(os.sep)}_{meta_hash}_{label_src}_{src_hash}"
    hashed = hashlib.md5(identifier.encode("utf-8")).hexdigest()[:8]
    cache_file = os.path.join(CACHE_DIR, f"{hashed}.npy")

    if os.path.exists(cache_file):
        print(f"reading HEEDB overread mask from on-disk cache: {cache_file}")
        loaded = np.load(cache_file, allow_pickle=True).item()
        mask = loaded["mask"]
        n_blank, n_missing = loaded["n_blank"], loaded["n_missing"]
        print("replaying below stats from cache:")
    else:
        print(f"reading HEEDB physician overreads from source: {heedb_path}")
        mgb_codes, euh_codes = get_fname_to_code(heedb_path, label_src)
        mask = np.zeros(len(meta), dtype=bool)
        n_blank, n_missing = 0, 0
        for meta_idx, codes in enumerate(
            tqdm(
                lookup_codes(meta, mgb_codes, euh_codes),
                total=len(meta),
                desc="Checking physician overreads",
            )
        ):
            if codes is None:
                n_missing += 1
            elif not isinstance(codes, str) or codes.strip() == "":
                # pandas reads an empty field as NaN
                n_blank += 1
            else:
                mask[meta_idx] = True
        os.makedirs(CACHE_DIR, exist_ok=True)
        saved = {"mask": mask, "n_blank": n_blank, "n_missing": n_missing}
        atomic_write(cache_file, lambda f: np.save(f, saved))  # type: ignore
        print(f"saved HEEDB overread mask to on-disk cache: {cache_file}")
    print(
        f"Of {len(meta)} ECGs, keeping {mask.sum()} with a physician overread, dropping "
        f"{n_blank} with a blank overread and {n_missing} not found in the diagnoses table"
    )
    print("===================================================")
    return mask


def hash_path_list(paths: pd.Series, chunk=100_000) -> str:
    m = hashlib.sha256()
    for i in range(0, len(paths), chunk):
        m.update("\0".join(paths.iloc[i : i + chunk]).encode())
        m.update(b"\0")
    return m.hexdigest()


def source_fingerprint(paths: list) -> str:
    # size and mtime of source files, so a new HEEDB release at the same path isn't served
    # stale caches
    parts = []
    for p in paths:
        st = os.stat(p)
        parts.append(f"{p}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.md5("\0".join(parts).encode("utf-8")).hexdigest()[:8]


def label_src_fingerprint(heedb_path: str, label_src: LABEL_SRC_T) -> str:
    label_csv, _label_col = LABEL_SRC_MAPPING[label_src]
    return source_fingerprint(
        [
            os.path.join(heedb_path, subdir, "12SL_diagnoses", label_csv)
            for subdir in ("I0001", "I0006")
        ]
    )


def waveform_cache_key(df: pd.DataFrame) -> str:
    # the exact rows in order, their normalization and the loader code, so a metadata,
    # filtering or preprocessing change can't silently misalign or alter cached waveforms
    m = hashlib.sha256()
    m.update(hash_path_list(df["full_path"]).encode())
    m.update(hash_path_list(df["source"]).encode())
    m.update(json.dumps([STREAM_STATS, heedb_lead_order]).encode())
    m.update(source_code_hash(_streaming_ecgs).encode())
    return m.hexdigest()[:16]


def source_code_hash(obj) -> str:
    # hash of the parsed source of a function or module, so comment and formatting edits
    # don't invalidate caches (the 500 GB waveform cache in particular)
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    return hashlib.md5(ast.dump(tree).encode("utf-8")).hexdigest()[:8]
