from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from wfdb import rdsamp

from ..defines import (
    MIMIC_CLIPPED_MEANS,
    MIMIC_CLIPPED_STDS,
    MIMIC_LOWERS,
    MIMIC_TARGETS,
    MIMIC_UPPERS,
    SPLIT_T,
)
from ._base_ecg_dataset import BaseECGDataset, load_cached_data


class MimicECGDataset(BaseECGDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        if label_subset is not None:
            raise NotImplementedError(
                f"label_subset not yet supported for {type(self.__name__)}"
            )
        _path = Path(dataset_path)
        df = pd.read_csv(_path / "ed-ecgs.csv")
        df = df[df["split"] == split]

        self.patient_ids = torch.as_tensor(df["subject_id"].to_numpy())
        self.ecg_ids = torch.as_tensor(df["study_id"].to_numpy())
        self.labels = torch.as_tensor(df[MIMIC_TARGETS].to_numpy(), dtype=torch.long)

        def load_transform_data_fn() -> torch.Tensor:
            data = []
            source_freq = 500
            for f in df["file_name"]:
                signal, meta = rdsamp(_path / f)
                assert signal is not None
                assert meta["fs"] == source_freq
                assert signal.shape == (5000, 12)
                data.append(signal)
            X = np.array(data)  # (N, 10 * source_freq, 12)

            # clip and normalize using stats derived over train set
            X = np.clip(X, MIMIC_LOWERS, MIMIC_UPPERS)
            X = (X - MIMIC_CLIPPED_MEANS) / MIMIC_CLIPPED_STDS

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
        assert self.patient_ids.shape[0] == self.labels.shape[0]
