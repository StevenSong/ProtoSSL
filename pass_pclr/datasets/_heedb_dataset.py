import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pass_pclr.datasets import BaseECGDataset, StreamingECGWaveforms
from pass_pclr.defines import (
    HEEDB_CLIPPED_MEANS,
    HEEDB_CLIPPED_STDS,
    HEEDB_LOWERS,
    HEEDB_TARGETS,
    HEEDB_UPPERS,
    SPLIT_T,
)


class HeedbECGDataset(BaseECGDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
    ):
        _path = Path(dataset_path)
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
        df = df.reset_index()[["ecg_id", "patient_id", "age", "sex", "year", "fpath"]]

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
        self.labels = torch.as_tensor(get_heedb_labels(dataset_path, df))
        self._df = df

        wfdb_paths = [_path / "I0001/WFDB" / f[1:] for f in df["fpath"]]
        self.waveforms = StreamingECGWaveforms(
            wfdb_paths=wfdb_paths,
            sampling_rate=sampling_rate,
            per_lead_lowerbound=HEEDB_LOWERS,
            per_lead_upperbound=HEEDB_UPPERS,
            per_lead_mean=HEEDB_CLIPPED_MEANS,
            per_lead_std=HEEDB_CLIPPED_STDS,
        )

        assert self.patient_ids.shape[0] == self.waveforms.shape[0]
        assert self.patient_ids.shape[0] == self.ecg_ids.shape[0]
        assert self.patient_ids.shape[0] == self.labels.shape[0]


FNAME_TO_CODE = None


def get_heedb_labels(heedb_path: str, meta: pd.DataFrame) -> np.ndarray:
    print("=================make_heedb_labels=================")
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
    code_to_label = {c: k for k, cs in HEEDB_TARGETS.items() for c in cs}
    label_to_idx = {k: i for i, k in enumerate(HEEDB_TARGETS)}

    data = np.zeros((len(meta), len(HEEDB_TARGETS)), dtype=np.long)
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
