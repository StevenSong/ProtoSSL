import hashlib
import os
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
)
from ._base_dataset import BaseTSDataset, load_cached_data, validate_label_subset
from .streaming_loaders import StreamingECGWaveforms

# fmt: off
LABEL_SRC_MAPPING = {
    "original_muse":      ("diagnoses_acquisition.csv", "codes_software"),
    "original_physician": ("diagnoses_acquisition.csv", "codes_physician"),
    "new_muse":           ("diagnoses_v24.csv",         "codes"),
}
# fmt: on
LABEL_SRC_T = Literal["original_muse", "original_physician", "new_muse"]
HEEDB_SPLIT_T = Literal["by-year", "by-label"]
BY_LABEL_SPLIT_HASH = "373878c0ce8ff856a22b738dbbac4830476568b8c482b5d597ab9624ae3bacaf"  # pragma: allowlist secret

heedb_lead_order = [l.lower() for l in HEEDB_LEAD_ORDER]
standard_lead_order = [l.lower() for l in STANDARD_LEAD_ORDER]
assert all([c == s for c, s in zip(heedb_lead_order, standard_lead_order)])

# DANGER: set this to load the entire waveform database into memory
# DANGER: the initial cache should be done prior to any jobs
HIGH_MEMORY = os.environ.get("HIGH_MEMORY", None) is not None

# global cache variables
FULL_META = None
MGB_FNAME_TO_CODE: dict[str, dict[str, str]] = dict()
EUH_FNAME_TO_CODE: dict[str, dict[str, str]] = dict()


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
    ):
        full_df = get_heedb_metadata(dataset_path, heedb_split_type=heedb_split_type)

        df = full_df[full_df["split"] == split].reset_index(drop=True)
        self.source_ids = torch.as_tensor(df["patient_id"].to_numpy())
        self.sample_ids = torch.as_tensor(df["ecg_id"].to_numpy())
        self.labels = torch.as_tensor(
            get_heedb_labels(dataset_path, df, label_subset, label_src)
        )
        self._df = df

        streaming_ecgs = StreamingECGWaveforms(
            wfdb_paths=list(df["full_path"]),
            sampling_rate=sampling_rate,
            per_lead_lowerbound={
                "mgb": HEEDB_MGB_LOWERS,
                "emory": HEEDB_EUH_LOWERS,
            },
            per_lead_upperbound={
                "mgb": HEEDB_MGB_UPPERS,
                "emory": HEEDB_EUH_UPPERS,
            },
            per_lead_mean={
                "mgb": HEEDB_MGB_CLIPPED_MEANS,
                "emory": HEEDB_EUH_CLIPPED_MEANS,
            },
            per_lead_std={
                "mgb": HEEDB_MGB_CLIPPED_STDS,
                "emory": HEEDB_EUH_CLIPPED_STDS,
            },
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

            def load_transform_data_fn() -> torch.Tensor:
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
                split=(
                    split
                    if heedb_split_type == "by-year"
                    else f"{split}-{heedb_split_type}"
                ),  # type: ignore
                sampling_rate=sampling_rate,
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
    global FULL_META
    if FULL_META is not None:
        return FULL_META.copy()

    print("================get_heedb_metadata=================")
    print(f"using {heedb_split_type} splits")

    identifier = f"HEEDB_{heedb_path.rstrip(os.sep)}_{heedb_split_type}"
    hashed = hashlib.md5(identifier.encode("utf-8")).hexdigest()[:8]
    cache_file = os.path.join(CACHE_DIR, f"{hashed}.csv")
    if os.path.exists(cache_file):
        print(f"reading HEEDB metadata from on-disk cache: {cache_file}")
        df = pd.read_csv(cache_file)
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
            # to splits which preserve independent label prevalence (need to check about cooccurrence)
            n = int(len(df) * 0.05)
            rng = np.random.default_rng(42)
            val_test_idxs = rng.choice(len(df), size=n * 2, replace=False)
            digest = hashlib.sha256(val_test_idxs.tobytes()).hexdigest()
            if digest != BY_LABEL_SPLIT_HASH:
                raise ValueError(
                    "Randomly generated split for preserving label prevalence has changed!"
                )

            df.loc[val_test_idxs[:n], "split"] = "val"
            df.loc[val_test_idxs[n:], "split"] = "test"
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
            full_paths.append(p)
        df["full_path"] = full_paths
        df.to_csv(cache_file, index=False)
        print(f"saved HEEDB metadata to on-disk cache: {cache_file}")

    FULL_META = df.copy()
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

    def make_fname_to_code(institution, label_csv, label_col) -> dict[str, str]:
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
                    # v24 file names are prefixed with '.' and suffixed with '.hea\n'
                    df["FileName"].str.strip(".").str.strip(".hea\n"),
                    desc=f"Creating {institution} fname to code mapping",
                ),
                df[label_col],
            )
        }

    meta_hash = hash_path_list(meta["fpath"])
    targets_hash = hashlib.md5("\0".join(targets).encode("utf-8")).hexdigest()[:8]
    identifier = f"HEEDB_labels_{heedb_path.rstrip(os.sep)}_{meta_hash}_{label_src}_{targets_hash}"
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
        _label_csv, _label_col = LABEL_SRC_MAPPING[label_src]
        if label_src not in MGB_FNAME_TO_CODE:
            MGB_FNAME_TO_CODE[label_src] = make_fname_to_code(
                "mgb", _label_csv, _label_col
            )
        if label_src not in EUH_FNAME_TO_CODE:
            EUH_FNAME_TO_CODE[label_src] = make_fname_to_code(
                "emory", _label_csv, _label_col
            )
        code_to_label = {c: k for k, cs in targets.items() for c in cs}
        label_to_idx = {k: i for i, k in enumerate(targets)}

        data = np.zeros((len(meta), len(targets)), dtype=np.long)
        count = 0
        for meta_idx, (fname, institution) in enumerate(
            zip(
                tqdm(meta["fpath"], desc="Converting code string to labels"),
                meta["source"],
            )
        ):
            if institution == "mgb":
                codes = MGB_FNAME_TO_CODE[label_src].get(fname, "MISSING")
            elif institution == "emory":
                codes = EUH_FNAME_TO_CODE[label_src].get(fname, "MISSING")
            else:
                raise ValueError(f"Unknown institution: {institution}")
            if codes == "MISSING":
                # separate missing vs empty (below)
                continue
            count += 1
            if isinstance(codes, float) and np.isnan(codes):
                continue
            assert isinstance(codes, str)
            for code in codes.split(","):
                # convert 12SL code to label
                label = code_to_label.get(int(code), -1)
                if label == -1:
                    continue
                # convert label to column idx
                label_idx = label_to_idx.get(label, -1)  # type: ignore
                if label_idx == -1:
                    continue
                data[meta_idx, label_idx] = 1

        n_ecgs, n_matched = len(meta), count
        n_annots = len(MGB_FNAME_TO_CODE[label_src]) + len(EUH_FNAME_TO_CODE[label_src])
        np.save(
            cache_file,
            {  # type: ignore
                "data": data,
                "n_ecgs": n_ecgs,
                "n_annots": n_annots,
                "n_matched": n_matched,
            },
        )
        print(f"saved HEEDB labels to on-disk cache: {cache_file}")
    print(
        f"Of {n_ecgs} ECGs and {n_annots} annotations, {n_matched} matched "
        f"({n_ecgs-n_matched} ECG annotations were missing and filled with 0s)"
    )
    print("===================================================")
    return data


def hash_path_list(paths: pd.Series, chunk=100_000) -> str:
    m = hashlib.sha256()
    for i in range(0, len(paths), chunk):
        m.update("\0".join(paths.iloc[i : i + chunk]).encode())
        m.update(b"\0")
    return m.hexdigest()
