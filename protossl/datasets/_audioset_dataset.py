from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..defines import SPLIT_T
from ._base_ecg_dataset import BaseTSDataset
from .streaming_loaders import StreamingAudioWaveforms


class AudioSetDataset(BaseTSDataset):
    def __init__(
        self,
        *,
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

        class_csv = _path / "audioset_train" / "class_labels_indices.csv"
        class_df = pd.read_csv(class_csv)

        # fixed class ordering from AudioSet metadata
        self.label_names = class_df["mid"].tolist()
        mid_to_idx = {mid: i for i, mid in enumerate(self.label_names)}

        if split in ["train", "val"]:
            full_df = pd.read_csv(_path / "audioset_train" / "train.csv")
            wav_dir = _path / "audioset_train" / "train_wav"

            # deterministic split by unique YTID
            unique_ytids = np.array(sorted(full_df["YTID"].unique()))
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(unique_ytids))

            val_frac = 0.1
            n_val = int(round(len(unique_ytids) * val_frac))
            val_ytids = set(unique_ytids[perm[:n_val]])
            train_ytids = set(unique_ytids[perm[n_val:]])

            if split == "train":
                df = full_df[full_df["YTID"].isin(train_ytids)].reset_index(drop=True)
            else:
                df = full_df[full_df["YTID"].isin(val_ytids)].reset_index(drop=True)

        elif split == "test":
            df = pd.read_csv(_path / "audioset_valid" / "valid.csv")
            wav_dir = _path / "audioset_valid" / "valid_wav"

        else:
            raise ValueError(f"Unknown split: {split}")

        # numeric IDs for repo compatibility
        ytid_codes, _ = pd.factorize(df["YTID"], sort=True)
        self.source_ids = torch.as_tensor(ytid_codes, dtype=torch.long)
        self.sample_ids = torch.arange(len(df), dtype=torch.long)

        # multi-hot labels
        labels = torch.zeros((len(df), len(self.label_names)), dtype=torch.long)
        for i, raw in enumerate(df["positive_labels"]):
            if pd.isna(raw):
                continue
            for mid in str(raw).split(","):
                mid = mid.strip()
                if mid in mid_to_idx:
                    labels[i, mid_to_idx[mid]] = 1
        self.labels = labels

        wav_paths = [wav_dir / f"{ytid}.wav" for ytid in df["YTID"]]

        # keep only rows with existing wavs
        keep = [p.exists() for p in wav_paths]
        if not all(keep):
            wav_paths = [p for p, k in zip(wav_paths, keep) if k]
            self.source_ids = self.source_ids[keep]
            self.sample_ids = self.sample_ids[keep]
            self.labels = self.labels[keep]

        self.waveforms = StreamingAudioWaveforms(
            wav_paths=wav_paths,
            sampling_rate=sampling_rate,
            clip_seconds=10.0,
        )

        assert self.source_ids.shape[0] == self.waveforms.shape[0]
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]
        assert self.source_ids.shape[0] == self.labels.shape[0]
