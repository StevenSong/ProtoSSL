from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from wfdb import rdsamp

from ..defines import (
    CINC_CLIPPED_MEANS,
    CINC_CLIPPED_STDS,
    CINC_LEAD_ORDER,
    CINC_LOWERS,
    CINC_TARGETS,
    CINC_UPPERS,
    SPLIT_T,
    STANDARD_LEAD_ORDER,
)
from ._base_dataset import BaseTSDataset, load_cached_data, validate_label_subset

cinc_lead_order = [l.lower() for l in CINC_LEAD_ORDER]
standard_lead_order = [l.lower() for l in STANDARD_LEAD_ORDER]
assert all([c == s for c, s in zip(cinc_lead_order, standard_lead_order)])


class CincECGDataset(BaseTSDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        targets = CINC_TARGETS
        if label_subset is not None:
            validate_label_subset(label_subset, CINC_TARGETS)
            targets = label_subset
        _path = Path(dataset_path)
        df = pd.read_csv(_path / "georgia.csv")
        df = df[df["split"] == split]

        self.source_ids = torch.as_tensor(df["patient_id"].to_numpy())
        self.sample_ids = torch.as_tensor(df["ecg_id"].to_numpy())
        self.labels = torch.as_tensor(df[targets].to_numpy(), dtype=torch.long)

        def load_transform_data_fn() -> torch.Tensor:
            data = []
            source_freq = 500
            for f in df["filename"]:
                signal, meta = rdsamp(_path / f)
                assert signal is not None
                assert meta["fs"] == source_freq
                assert signal.shape == (5000, 12)
                lead_order = [l.lower() for l in meta["sig_name"]]
                assert all([c == l for c, l in zip(cinc_lead_order, lead_order)])
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

        assert self.source_ids.shape[0] == self.waveforms.shape[0]
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]
        assert self.labels is None or self.source_ids.shape[0] == self.labels.shape[0]
