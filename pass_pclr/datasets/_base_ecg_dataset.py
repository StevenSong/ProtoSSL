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
    identifier = f"{dataset_path}_{split}_{sampling_rate}"
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

    # load cached data
    X = torch.load(cache_file)
    print(f"Cached dataset loaded from {cache_file}")
    print("===================================================")
    return X


class StreamingECGWaveforms:
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
    ):
        """
        :param dataset_path: path to dataset directory
        :type dataset_path: str
        :param split: one of `"train"`, `"val"`, or `"test"`
        :type split: str
        :param sampling_rate: target sampling rate to sample waveform at
        :type split: int
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
