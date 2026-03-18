from fractions import Fraction

import numpy as np
import torch
from scipy.signal import resample_poly
from wfdb import rdsamp

from ._streaming_base import StreamingWaveformsBase


class StreamingECGWaveforms(StreamingWaveformsBase):
    def __init__(
        self,
        *,  # enforce kwargs
        wfdb_paths: list[str],
        sampling_rate: int,
        per_lead_lowerbound: list[float],
        per_lead_upperbound: list[float],
        per_lead_mean: list[float],
        per_lead_std: list[float],
    ):
        self.wfdb_paths = wfdb_paths
        self.sampling_rate = sampling_rate
        self.per_lead_lowerbound = per_lead_lowerbound
        self.per_lead_upperbound = per_lead_upperbound
        self.per_lead_mean = per_lead_mean
        self.per_lead_std = per_lead_std
        print("===============StreamingECGWaveforms===============")
        print("Using streaming ECG waveforms, will load and transform data on the fly")
        print("===================================================")

    def __getitem__(self, i: int) -> torch.Tensor:
        fpath = self.wfdb_paths[i]

        # assume 10-sec 12-lead ECGs
        x: np.ndarray = rdsamp(fpath)[0]  # type: ignore

        # make shape (T[imesteps], L[eads]) - easier for broadcasting ops
        if x.shape[0] == 12:
            x = x.mT  # (T, L)

        # normalize
        x = np.clip(x, self.per_lead_lowerbound, self.per_lead_upperbound)
        x = (x - self.per_lead_mean) / self.per_lead_std

        # resample to target freq
        source_freq = x.shape[0] // 10
        if self.sampling_rate != source_freq:
            resample_frac = Fraction(
                numerator=self.sampling_rate,
                denominator=source_freq,
            ).limit_denominator(100)
            x = resample_poly(
                x,
                up=resample_frac.numerator,
                down=resample_frac.denominator,
                axis=0,
            )  # (10 * sampling_rate, 12)

        x = x.astype(np.float32)
        return torch.as_tensor(x).mT  # (L, T) - shape expected by BaseECGDataset

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.wfdb_paths), self.sampling_rate * 10)
