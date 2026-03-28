import hashlib
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from wfdb import rdsamp

from ..defines import CACHE_DIR, SPLIT_T

INVALIDATE_CACHE = os.environ.get("INVALIDATE_CACHE") == "yes im sure"
if INVALIDATE_CACHE:
    print("==================INVALIDATE_CACHE=================")
    print("INVALIDATE_CACHE env var set, deleting entire cache")
    print("===================================================")
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)


def load_cached_data(
    *,  # enforce kwargs
    load_transform_data_fn: Callable[[], torch.Tensor],
    dataset_path: str,
    split: SPLIT_T,
    sampling_rate: int,
):
    # loading and processing can be time intensive, so cache transformed data if it doesn't already exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # use a short hash of the source dataest path to distinguish different sources (e.g. if using different dataset subsets)
    identifier = f"{dataset_path.rstrip(os.sep)}_{split}_{sampling_rate}"
    hashed = hashlib.md5(identifier.encode("utf-8")).hexdigest()[:8]
    cache_file = os.path.join(CACHE_DIR, f"{hashed}.pt")

    print("=================load_cached_data==================")
    print(f"Dataset parameters: ({dataset_path}, {split}, {sampling_rate}Hz)")
    if not os.path.exists(cache_file):
        print("Cache file not found, loading/transforming from source")

        # all params and processing should be wrapped in this loader function
        X = load_transform_data_fn()
        X = X.float()

        # cache data
        torch.save(X, cache_file)
        print(f"Dataset cached to {cache_file}")
        del X

    # load cached data
    X = torch.load(cache_file)
    print(f"Cached dataset loaded from {cache_file}")
    print("===================================================")
    return X


def validate_label_subset(
    label_subset: list[str],
    labels: list[str],
):
    _labels = set(labels)
    for label in label_subset:
        if label not in _labels:
            raise ValueError(f"'{label}' is not in the current label set")


class StreamingECGWaveforms:
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
        if verbose:
            print("===============StreamingECGWaveforms===============")
            print(
                "Using streaming ECG waveforms, will load and transform data on the fly"
            )
            print("===================================================")

    def __getitem__(self, i: int) -> torch.Tensor:
        fpath = self.wfdb_paths[i]

        # assume 10-sec 12-lead ECGs
        x: np.ndarray = rdsamp(fpath)[0]  # type: ignore

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

    def __len__(self) -> int:
        return len(self.wfdb_paths)


class BaseECGDataset(Dataset, ABC):
    patient_ids: torch.Tensor  # (N,), N = n_samples
    ecg_ids: torch.Tensor  # (N,), N = n_samples
    waveforms: (
        torch.Tensor | StreamingECGWaveforms
    )  # (N, L, T), L = n_leads, T = n_timesteps
    labels: torch.Tensor | None  # (N, C), C = n_binary_labels

    @abstractmethod
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        """
        :param dataset_path: path to dataset directory
        :type dataset_path: str
        :param split: one of `"train"`, `"val"`, or `"test"`
        :type split: str
        :param sampling_rate: target sampling rate to sample waveform at
        :type split: int
        :param label_subset: optional subset of labels to use
        :type split: list[str] | None
        """
        pass

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ret = {
            "waveform": self.waveforms[i],  # (L, T)
            "patient_id": self.patient_ids[i],  # (,)
            "ecg_id": self.ecg_ids[i],  # (,)
        }
        if self.labels is not None:
            ret["label"] = self.labels[i]  # (C,)
        return ret

    def __len__(self) -> int:
        return self.waveforms.shape[0]

    def get_label_cooccurrence(self) -> torch.Tensor:
        # jaccard similarity
        if self.labels is None:
            raise ValueError("This dataset does not have any labels!")
        cooc_counts = self.labels.T @ self.labels  # (L, L)
        per_label_count = self.labels.sum(dim=0)  # (L,)
        union = (
            per_label_count.unsqueeze(1)  # (L, 1)
            + per_label_count.unsqueeze(0)  # (1, L)
            - cooc_counts  # don't double count
        )
        return cooc_counts / (union + 1e-10)

    def get_label_weights(self) -> torch.Tensor:
        if self.labels is None:
            raise ValueError("This dataset does not have any labels!")
        n_samples = self.labels.shape[0]
        per_label_count = self.labels.sum(dim=0)
        return (n_samples - per_label_count) / per_label_count
