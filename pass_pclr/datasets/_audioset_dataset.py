from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import resample_poly

from ..defines import SPLIT_T
from ._base_ecg_dataset import BaseECGDataset


class StreamingAudioWaveforms:
    def __init__(
        self,
        *,
        wav_paths: list[Path],
        sampling_rate: int,
        clip_seconds: float = 10.0,
    ):
        self.wav_paths = wav_paths
        self.sampling_rate = sampling_rate
        self.clip_seconds = clip_seconds
        self.target_len = int(sampling_rate * clip_seconds)
        print("================StreamingAudioWaveforms================")
        print("Loading and transforming audio data.")
        print("=======================================================")

    def __getitem__(self, i: int) -> torch.Tensor:
        fpath = self.wav_paths[i]
        x, source_sr = torchaudio.load(str(fpath))  # (C, T)

        # convert to mono
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)  # (1, T)

        # resample if needed
        if source_sr != self.sampling_rate:
            x_np = x.squeeze(0).numpy()
            frac = Fraction(
                numerator=self.sampling_rate,
                denominator=source_sr,
            ).limit_denominator(100)
            x_np = resample_poly(
                x_np,
                up=frac.numerator,
                down=frac.denominator,
                axis=0,
            )
            x = torch.as_tensor(x_np, dtype=torch.float32).unsqueeze(0)
        else:
            x = x.float()

        # crop or pad to fixed 10-second length
        cur_len = x.shape[1]
        if cur_len > self.target_len:
            x = x[:, : self.target_len]
        elif cur_len < self.target_len:
            x = F.pad(x, (0, self.target_len - cur_len))

        # simple peak normalization
        peak = x.abs().max()
        if peak > 0:
            x = x / peak

        return x  # (1, T)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.wav_paths), 1, self.target_len)


class AudioSetDataset(BaseECGDataset):
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
        self.patient_ids = torch.as_tensor(ytid_codes, dtype=torch.long)
        self.ecg_ids = torch.arange(len(df), dtype=torch.long)

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
            self.patient_ids = self.patient_ids[keep]
            self.ecg_ids = self.ecg_ids[keep]
            self.labels = self.labels[keep]

        self.waveforms = StreamingAudioWaveforms(
            wav_paths=wav_paths,
            sampling_rate=sampling_rate,
            clip_seconds=10.0,
        )

        assert self.patient_ids.shape[0] == self.waveforms.shape[0]
        assert self.patient_ids.shape[0] == self.ecg_ids.shape[0]
        assert self.patient_ids.shape[0] == self.labels.shape[0]
