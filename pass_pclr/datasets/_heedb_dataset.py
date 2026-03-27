import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..defines import (
    HEEDB_CLIPPED_MEANS,
    HEEDB_CLIPPED_STDS,
    HEEDB_LOWERS,
    HEEDB_TARGETS,
    HEEDB_UPPERS,
    SPLIT_T,
)
from ._base_ecg_dataset import (
    BaseECGDataset,
    StreamingECGWaveforms,
    load_cached_data,
    validate_label_subset,
)

FULL_META = None
HIGH_MEMORY = os.environ.get("HIGH_MEMORY", None) is not None


class HeedbECGDataset(BaseECGDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        _path = Path(dataset_path)
        global FULL_META
        if FULL_META is None:
            df = pd.read_csv(
                _path / "I0001/metadata/metadata.csv",
                usecols=["BDSPPatientID", "SexDSC", "AgeAtAcquisition", "FileName"],
            )
            df["AgeAtAcquisition"] = df["AgeAtAcquisition"] / 365.2425
            df = df[(df["AgeAtAcquisition"] >= 18) & (df["SexDSC"].notna())]
            df = df.rename(
                columns={
                    "BDSPPatientID": "patient_id",
                    "SexDSC": "sex",
                    "AgeAtAcquisition": "age",
                    "FileName": "fpath",
                }
            )
            df.index.name = "ecg_id"
            df["year"] = df["fpath"].str[1:].str.split("/").str[1].astype(int)
            df = df.reset_index()[
                ["ecg_id", "patient_id", "age", "sex", "year", "fpath"]
            ]
            FULL_META = df.copy()
        else:
            df = FULL_META.copy()

        if split == "train":
            mask = ~df["year"].isin([2021, 2022])
        elif split == "val":
            mask = df["year"] == 2021
        elif split == "test":
            mask = df["year"] == 2022
        else:
            raise ValueError(f"Unknown split: {split}")

        df = df[mask].reset_index(drop=True)
        self.patient_ids = torch.as_tensor(df["patient_id"].to_numpy())
        self.ecg_ids = torch.as_tensor(df["ecg_id"].to_numpy())
        self.labels = torch.as_tensor(get_heedb_labels(dataset_path, df, label_subset))
        self._df = df

        wfdb_paths = [_path / "I0001/WFDB" / f[1:] for f in df["fpath"]]
        streaming_ecgs = StreamingECGWaveforms(
            wfdb_paths=wfdb_paths,
            sampling_rate=sampling_rate,
            per_lead_lowerbound=HEEDB_LOWERS,
            per_lead_upperbound=HEEDB_UPPERS,
            per_lead_mean=HEEDB_CLIPPED_MEANS,
            per_lead_std=HEEDB_CLIPPED_STDS,
            verbose=not HIGH_MEMORY,
        )

        if not HIGH_MEMORY:
            self.waveforms = streaming_ecgs
        else:

            def load_transform_data_fn() -> torch.Tensor:
                print("WARNING:")
                print(
                    "WARNING: ABOUT TO LOAD ENTIRE HEEDB WAVEFORM MATRIX INTO MEMORY TO CACHE"
                )
                print(
                    "WARNING: THIS USES A DATALOADER AND SHOULD NOT BE DONE INSIDE A TRAINING JOB"
                )
                print("WARNING:")
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

        assert self.patient_ids.shape[0] == self.waveforms.shape[0]
        assert self.patient_ids.shape[0] == self.ecg_ids.shape[0]
        assert self.patient_ids.shape[0] == self.labels.shape[0]


FNAME_TO_CODE = None


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
    global FNAME_TO_CODE
    if FNAME_TO_CODE is None:
        df = pd.read_csv(
            os.path.join(heedb_path, "I0001/12SL_diagnoses/diagnoses_acquisition.csv")
        )
        FNAME_TO_CODE = {
            fname: code_str
            for fname, code_str in zip(
                tqdm(df["FileName"], desc="Creating fname to code mapping"),
                df["codes_physician"],
            )
        }
    code_to_label = {c: k for k, cs in targets.items() for c in cs}
    label_to_idx = {k: i for i, k in enumerate(targets)}

    data = np.zeros((len(meta), len(targets)), dtype=np.long)
    count = 0
    for meta_idx, fname in enumerate(
        tqdm(meta["fpath"], desc="Converting code string to labels")
    ):
        codes = FNAME_TO_CODE.get(fname, "MISSING")
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
        f"Of {len(meta)} ECGs and {len(FNAME_TO_CODE)} annotations, {count} matched ({len(meta)-count} ECG annotations were missing and filled with 0s)"
    )
    print("===================================================")
    return data
