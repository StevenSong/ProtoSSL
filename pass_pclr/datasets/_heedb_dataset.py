from pathlib import Path

import pandas as pd
import torch

from ..defines import (
    HEEDB_CLIPPED_MEANS,
    HEEDB_CLIPPED_STDS,
    HEEDB_LOWERS,
    HEEDB_UPPERS,
    SPLIT_T,
)
from . import BaseECGDataset
from .streaming_loaders import StreamingECGWaveforms


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
        df["FileName"] = df["FileName"].str[1:]
        df = df.rename(
            columns={
                "BDSPPatientID": "patient_id",
                "SexDSC": "sex",
                "AgeAtAcquisition": "age",
                "FileName": "fpath",
            }
        )
        df.index.name = "ecg_id"
        df["year"] = df["fpath"].str.split("/").str[1].astype(int)
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
        self.labels = None

        wfdb_paths = [_path / "I0001/WFDB" / f for f in df["fpath"]]
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
        assert self.labels is None or self.patient_ids.shape[0] == self.labels.shape[0]
