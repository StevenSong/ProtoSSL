import hashlib
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch.utils.data import Dataset

from ..defines import CACHE_DIR, SPLIT_T
from .streaming_loaders import StreamingWaveformsBase

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


class BaseTSDataset(Dataset, ABC):
    source_ids: torch.Tensor  # (N,), N = n_samples
    sample_ids: torch.Tensor  # (N,), N = n_samples
    waveforms: torch.Tensor | StreamingWaveformsBase  # (N, n_leads, n_timesteps)
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
            "source_id": self.source_ids[i],  # (,)
            "sample_id": self.sample_ids[i],  # (,)
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
