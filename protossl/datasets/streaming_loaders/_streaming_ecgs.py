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
        per_lead_lowerbound: list[float] | dict[str, list[float]],
        per_lead_upperbound: list[float] | dict[str, list[float]],
        per_lead_mean: list[float] | dict[str, list[float]],
        per_lead_std: list[float] | dict[str, list[float]],
        verbose: bool = True,
        stat_mapper: list[str] | None = None,
        expected_lead_order: list[str],
    ):
        # if a set of waveforms should be normalized using different stats,
        # each of the per-lead stats can be passed as a dictionary mapping some
        # indicator to the different sets of stats. additionally, stat_mapper
        # should be a list of those indicators annotating which samples should
        # get which set of stats.
        if stat_mapper is not None or isinstance(per_lead_lowerbound, dict):
            if stat_mapper is None or not isinstance(per_lead_lowerbound, dict):
                raise ValueError(
                    f"stat_mapper and mappable stats for normalization must be used together!"
                )
            if len(stat_mapper) != len(wfdb_paths):
                raise ValueError(
                    f"length of stat_mapper ({len(stat_mapper)}) must match length of wfdb_paths ({len(wfdb_paths)})"
                )
            if (
                not isinstance(per_lead_upperbound, dict)
                or not isinstance(per_lead_mean, dict)
                or not isinstance(per_lead_std, dict)
            ):
                raise ValueError(
                    f"if any stats are passed as mappable, then all stats must be as well"
                )
            indicators = set(stat_mapper)
            for per_lead_stat in [
                per_lead_lowerbound,
                per_lead_upperbound,
                per_lead_mean,
                per_lead_std,
            ]:
                if len(remainder := (indicators - set(per_lead_stat))) != 0:
                    raise ValueError(
                        f"some indicators ({remainder}) were present in stat_mapper but not in the mappings of the actual stats"
                    )

        self.wfdb_paths = wfdb_paths
        self.stat_mapper = stat_mapper
        self.sampling_rate = sampling_rate
        self.per_lead_lowerbound = per_lead_lowerbound
        self.per_lead_upperbound = per_lead_upperbound
        self.per_lead_mean = per_lead_mean
        self.per_lead_std = per_lead_std
        self.expected_lead_order = expected_lead_order
        if verbose:
            print("===============StreamingECGWaveforms===============")
            print(
                "Using streaming ECG waveforms, will load and transform data on the fly"
            )
            print("===================================================")

    def __getitem__(self, i: int) -> torch.Tensor:
        fpath = self.wfdb_paths[i]

        # assume 10-sec 12-lead ECGs
        _samp = rdsamp(fpath)
        x: np.ndarray = _samp[0]  # type: ignore
        meta = _samp[1]
        lead_order = [l.lower() for l in meta["sig_name"]]
        assert all([c == l for c, l in zip(self.expected_lead_order, lead_order)])

        # make shape (T[imesteps], L[eads]) - easier for broadcasting ops
        if x.shape[0] == 12:
            x = x.mT  # (T, L)

        # normalize
        if self.stat_mapper is not None:
            # this should already have been checked in init but just in case
            assert isinstance(self.per_lead_lowerbound, dict)
            assert isinstance(self.per_lead_upperbound, dict)
            assert isinstance(self.per_lead_mean, dict)
            assert isinstance(self.per_lead_std, dict)
            indicator = self.stat_mapper[i]
            curr_per_lead_lowerbound = self.per_lead_lowerbound[indicator]
            curr_per_lead_upperbound = self.per_lead_upperbound[indicator]
            curr_per_lead_mean = self.per_lead_mean[indicator]
            curr_per_lead_std = self.per_lead_std[indicator]
        else:
            # this should already have been checked in init but just in case
            assert not isinstance(self.per_lead_lowerbound, dict)
            assert not isinstance(self.per_lead_upperbound, dict)
            assert not isinstance(self.per_lead_mean, dict)
            assert not isinstance(self.per_lead_std, dict)
            curr_per_lead_lowerbound = self.per_lead_lowerbound
            curr_per_lead_upperbound = self.per_lead_upperbound
            curr_per_lead_mean = self.per_lead_mean
            curr_per_lead_std = self.per_lead_std

        x = np.clip(x, curr_per_lead_lowerbound, curr_per_lead_upperbound)
        x = (x - curr_per_lead_mean) / curr_per_lead_std

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
