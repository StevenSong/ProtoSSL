from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from wfdb import rdsamp

from pass_pclr.datasets import BaseECGDataset, load_cached_data
from pass_pclr.defines import SPLIT_T

VAL_FOLD = 9
TEST_FOLD = 10


class PtbxlECGDataset(BaseECGDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
    ):
        _path = Path(dataset_path)
        df = pd.read_csv(_path / "ptbxl_database.csv", index_col="ecg_id")
        if split == "train":
            mask = ~df["strat_fold"].isin({VAL_FOLD, TEST_FOLD})
        elif split == "val":
            mask = df["strat_fold"] == VAL_FOLD
        elif split == "test":
            mask = df["strat_fold"] == TEST_FOLD
        else:
            raise ValueError(f"Unknown split: {split}")
        df = df[mask]

        self.patient_ids = torch.as_tensor(df["patient_id"].astype(int).to_numpy())
        self.labels = None

        def load_transform_data_fn() -> torch.Tensor:
            if sampling_rate <= 100:
                source_freq = 100
                data = [rdsamp(_path / f) for f in df["filename_lr"]]
            else:
                source_freq = 500
                data = [rdsamp(_path / f) for f in df["filename_hr"]]
            X = np.array([signal for signal, meta in data])  # (N, 10 * source_freq, 12)

            # downsample to target frequency
            if sampling_rate != source_freq:
                resample_frac = Fraction(
                    numerator=sampling_rate,
                    denominator=source_freq,
                ).limit_denominator(100)
                X = resample_poly(
                    X,
                    up=resample_frac.numerator,
                    down=resample_frac.denominator,
                    axis=1,
                )  # (N, 10 * sampling_rate, 12)
            X = torch.as_tensor(X).mT  # (N, 12, 10 * sampling_rate)
            return X

        self.waveforms = load_cached_data(
            load_transform_data_fn=load_transform_data_fn,
            dataset_path=dataset_path,
            split=split,
            sampling_rate=sampling_rate,
        )

        assert self.patient_ids.shape[0] == self.waveforms.shape[0]
        assert self.labels is None or self.patient_ids.shape[0] == self.labels.shape[0]
