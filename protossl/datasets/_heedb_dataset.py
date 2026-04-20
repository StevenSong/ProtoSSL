import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..defines import (
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
from ._base_ecg_dataset import (
    BaseECGDataset,
    StreamingECGWaveforms,
    load_cached_data,
    validate_label_subset,
)

heedb_lead_order = [l.lower() for l in HEEDB_LEAD_ORDER]
standard_lead_order = [l.lower() for l in STANDARD_LEAD_ORDER]
assert all([c == s for c, s in zip(heedb_lead_order, standard_lead_order)])

# DANGER: set this to load the entire waveform database into memory
# DANGER: the initial cache should be done prior to any jobs
HIGH_MEMORY = os.environ.get("HIGH_MEMORY", None) is not None

# global cache variables
FULL_META = None
MGB_FNAME_TO_CODE = None
EUH_FNAME_TO_CODE = None


class HeedbECGDataset(BaseECGDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        df = get_heedb_metadata(dataset_path)

        df = df[df["split"] == split].reset_index(drop=True)
        self.source_ids = torch.as_tensor(df["source_id"].to_numpy())
        self.sample_ids = torch.as_tensor(df["sample_id"].to_numpy())
        self.labels = torch.as_tensor(get_heedb_labels(dataset_path, df, label_subset))
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
                split=split,
                sampling_rate=sampling_rate,
            )

        assert self.source_ids.shape[0] == self.waveforms.shape[0]
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]
        assert self.source_ids.shape[0] == self.labels.shape[0]


def get_heedb_metadata(heedb_path: str) -> pd.DataFrame:
    _path = Path(heedb_path)
    global FULL_META
    if FULL_META is not None:
        return FULL_META.copy()

    print("================get_heedb_metadata=================")
    print("reading HEEDB metadata")

    # read harvard data
    mgb = pd.read_csv(
        _path / "I0001/metadata/metadata.csv",
        usecols=["BDSPPatientID", "SexDSC", "AgeAtAcquisition", "FileName"],
    )
    mgb["AgeAtAcquisition"] = mgb["AgeAtAcquisition"] / 365.2425
    mgb = mgb[(mgb["AgeAtAcquisition"] >= 18) & (mgb["SexDSC"].notna())]
    mgb = mgb.rename(
        columns={
            "BDSPPatientID": "source_id",
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
            "BDSPPatientID": "source_id",
            "Sex": "sex",
            "AgeAtAcquisition": "age",
            "FileName": "fpath",
        }
    )
    assert (emory["source_id"].astype(int) == emory["source_id"]).all()
    emory["source_id"] = emory["source_id"].astype(int)

    # bad files on emory side
    emory_exclude = {"WFDB/2013/MUSE_20200225_081000_06000"}
    emory = emory[~emory["fpath"].isin(emory_exclude)]

    print("read Emory data")

    # join together
    mgb["source"] = "mgb"
    emory["source"] = "emory"
    assert len(set(mgb["source_id"]) & set(emory["source_id"])) == 0
    df = pd.concat([mgb, emory], ignore_index=True)  # MGB, then EUH
    df.index.name = "sample_id"
    df["year"] = df["fpath"].str[1:].str.split("/").str[1].astype(int)
    df = df.reset_index()[
        ["sample_id", "source_id", "age", "sex", "year", "source", "fpath"]
    ]

    # emory data ends in 2018 so val/test are all MGB data
    df["split"] = "no-split"
    df.loc[~df["year"].isin([2021, 2022]), "split"] = "train"
    df.loc[df["year"] == 2021, "split"] = "val"
    df.loc[df["year"] == 2022, "split"] = "test"

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

    FULL_META = df.copy()
    print("===================================================")
    return df


def get_heedb_labels(
    heedb_path: str,
    meta: pd.DataFrame,
    label_subset: list[str] | None = None,
) -> np.ndarray:
    print("=================make_heedb_labels=================")
    targets = HEEDB_TARGETS
    if label_subset is not None:
        validate_label_subset(label_subset, list(HEEDB_TARGETS))
        targets = {label: HEEDB_TARGETS[label] for label in label_subset}
    global MGB_FNAME_TO_CODE
    global EUH_FNAME_TO_CODE

    def make_fname_to_code(institution) -> dict[str, str]:
        if institution == "mgb":
            subdir = "I0001"
        elif institution == "emory":
            subdir = "I0006"
        else:
            raise ValueError(f"Unknown subdir for institution: {institution}")
        df = pd.read_csv(
            os.path.join(heedb_path, subdir, "12SL_diagnoses/diagnoses_acquisition.csv")
        )
        return {
            fname: code_str
            for fname, code_str in zip(
                tqdm(
                    df["FileName"], desc=f"Creating {institution} fname to code mapping"
                ),
                df["codes_physician"],
            )
        }

    if MGB_FNAME_TO_CODE is None:
        MGB_FNAME_TO_CODE = make_fname_to_code("mgb")
    if EUH_FNAME_TO_CODE is None:
        EUH_FNAME_TO_CODE = make_fname_to_code("emory")
    code_to_label = {c: k for k, cs in targets.items() for c in cs}
    label_to_idx = {k: i for i, k in enumerate(targets)}

    data = np.zeros((len(meta), len(targets)), dtype=np.long)
    count = 0
    for meta_idx, (fname, institution) in enumerate(
        zip(
            tqdm(meta["fpath"], desc="Converting code string to labels"), meta["source"]
        )
    ):
        if institution == "mgb":
            codes = MGB_FNAME_TO_CODE.get(fname, "MISSING")
        elif institution == "emory":
            codes = EUH_FNAME_TO_CODE.get(fname, "MISSING")
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
    print(
        f"Of {len(meta)} ECGs and {len(MGB_FNAME_TO_CODE) + len(EUH_FNAME_TO_CODE)} annotations, "
        f"{count} matched ({len(meta)-count} ECG annotations were missing and filled with 0s)"
    )
    print("===================================================")
    return data
