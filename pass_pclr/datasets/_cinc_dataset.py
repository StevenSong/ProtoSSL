from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from wfdb import rdsamp

from pass_pclr.datasets import BaseECGDataset, load_cached_data
from pass_pclr.defines import (
    CINC_CLIPPED_MEANS,
    CINC_CLIPPED_STDS,
    CINC_LOWERS,
    CINC_TARGETS,
    CINC_UPPERS,
    SPLIT_T,
)


class CincECGDataset(BaseECGDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
    ):
        _path = Path(dataset_path)
        df = pd.read_csv(_path / "georgia.csv")
        df = df[df["split"] == split]

        self.patient_ids = torch.as_tensor(df["patient_id"].to_numpy())
        self.ecg_ids = torch.as_tensor(df["ecg_id"].to_numpy())
        self.labels = torch.as_tensor(df[CINC_TARGETS].to_numpy(), dtype=torch.long)

        def load_transform_data_fn() -> torch.Tensor:
            data = []
            source_freq = 500
            for f in df["filename"]:
                signal, meta = rdsamp(_path / f)
                assert signal is not None
                assert meta["fs"] == source_freq
                assert signal.shape == (5000, 12)
                data.append(signal)
            X = np.array(data)  # (N, 10 * source_freq, 12)

            # clip and normalize using stats derived over train set
            X = np.clip(X, CINC_LOWERS, CINC_UPPERS)
            X = (X - CINC_CLIPPED_MEANS) / CINC_CLIPPED_STDS

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
        assert self.patient_ids.shape[0] == self.ecg_ids.shape[0]
        assert self.labels is None or self.patient_ids.shape[0] == self.labels.shape[0]
