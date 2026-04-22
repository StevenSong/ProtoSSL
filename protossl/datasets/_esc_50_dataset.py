import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..defines import ESC_50_TARGETS, SPLIT_T
from ._base_dataset import BaseTSDataset
from .streaming_loaders import StreamingAudioWaveforms

N_FOLDS = 5
DEFAULT_ESC_TEST_SPLIT = 0
ENV_VAR_NAME = "ESC_TEST_FOLD"


class Esc50Dataset(BaseTSDataset):
    def __init__(
        self,
        *,
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        """
        Data downloaded from: https://github.com/karolpiczak/ESC-50
        """
        if label_subset is not None:
            raise NotImplementedError(
                f"label_subset not yet supported for {type(self.__name__)}"
            )

        test_fold = int(os.environ.get(ENV_VAR_NAME, DEFAULT_ESC_TEST_SPLIT))
        val_fold = (test_fold + 1) % N_FOLDS
        train_folds = [i for i in range(N_FOLDS) if i not in {test_fold, val_fold}]
        print(f"======================ESC-50=======================")
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
        df = pd.read_csv(_path / "meta/esc50.csv")
        df["fold"] = df["fold"] - 1  # source folds in [1-5], reindex to [0-4]
        df["sample_id"] = np.arange(len(df))
        df = df[df["fold"].isin(use_folds)].reset_index(drop=True)
        self._df = df

        self.source_ids = torch.as_tensor(df["src_file"], dtype=torch.long)
        self.sample_ids = torch.as_tensor(df["sample_id"], dtype=torch.long)

        label_to_idx = {k: i for i, k in enumerate(ESC_50_TARGETS)}
        self.labels = torch.zeros((len(df), len(ESC_50_TARGETS)), dtype=torch.long)
        for i, l in enumerate(df["category"]):
            idx = label_to_idx[l]
            self.labels[i, idx] = 1

        wav_paths = (_path / "audio" / df["filename"]).to_list()
        self.waveforms = StreamingAudioWaveforms(
            wav_paths=wav_paths,
            sampling_rate=sampling_rate,
            clip_seconds=5.0,
        )

        assert self.source_ids.shape[0] == self.waveforms.shape[0]
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]
        assert self.source_ids.shape[0] == self.labels.shape[0]
