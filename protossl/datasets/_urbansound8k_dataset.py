import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..defines import SPLIT_T, URBANSOUND8K_TARGETS
from ._base_dataset import BaseTSDataset
from .streaming_loaders import StreamingAudioWaveforms

N_FOLDS = 10
DEFAULT_US8K_TEST_SPLIT = 0
ENV_VAR_NAME = "US8K_TEST_FOLD"


class UrbanSound8kDataset(BaseTSDataset):
    def __init__(
        self,
        *,
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        """
        Data downloaded from: https://urbansounddataset.weebly.com/urbansound8k.html
        """
        if label_subset is not None:
            raise NotImplementedError(
                f"label_subset not yet supported for {type(self.__name__)}"
            )

        test_fold = int(os.environ.get(ENV_VAR_NAME, DEFAULT_US8K_TEST_SPLIT))
        val_fold = (test_fold + 1) % N_FOLDS
        train_folds = [i for i in range(N_FOLDS) if i not in {test_fold, val_fold}]
        print(f"===================Urbansound8k====================")
        print(
            f"Using fold {test_fold} for test split and fold {val_fold} for val split."
        )
        print(
            f"To change this behavior, set the env var '{ENV_VAR_NAME}' to a value [0-{N_FOLDS-1}]."
        )
        print(f"NOTE: we reindex the source dataset folds to start from 0.")
        print(f"Fold used for val is automatically the fold after the one for test.")
        print(f"===================================================")

        if split == "train":
            use_folds = train_folds
        elif split == "val":
            use_folds = [val_fold]
        elif split == "test":
            use_folds = [test_fold]
        else:
            raise ValueError(f"Unknown fold {split}")

        _path = Path(dataset_path)
        df = pd.read_csv(_path / "metadata/UrbanSound8K.csv")
        df["folder"] = "fold" + df["fold"].astype(str)
        df["fold"] = df["fold"] - 1  # reindex folds from 0
        df["sample_id"] = np.arange(len(df))
        df = df[df["fold"].isin(use_folds)].reset_index(drop=True)
        self._df = df

        self.source_ids = torch.as_tensor(df["fsID"], dtype=torch.long)
        self.sample_ids = torch.as_tensor(df["sample_id"], dtype=torch.long)

        label_to_idx = {k: i for i, k in enumerate(URBANSOUND8K_TARGETS)}
        self.labels = torch.zeros((len(df), len(label_to_idx)), dtype=torch.long)
        for i, l in enumerate(df["class"]):
            idx = label_to_idx[l]
            self.labels[i, idx] = 1

        wav_paths = (_path / "audio" / df["folder"] / df["slice_file_name"]).to_list()
        self.waveforms = StreamingAudioWaveforms(
            wav_paths=wav_paths,
            sampling_rate=sampling_rate,
            clip_seconds=4.0,
        )

        assert self.source_ids.shape[0] == self.waveforms.shape[0]
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]
        assert self.source_ids.shape[0] == self.labels.shape[0]
