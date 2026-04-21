from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as HFDataset
from datasets.features import Audio
from scipy.signal import resample_poly

from ._streaming_base import StreamingWaveformsBase


class StreamingAudioWaveforms(StreamingWaveformsBase):
    def __init__(
        self,
        *,  # enforce kwargs
        wav_paths: list[Path] | None = None,
        hf_ds: HFDataset | None = None,
        sampling_rate: int,
        clip_seconds: float = 10.0,
    ):
        self.wav_paths = wav_paths
        self.hf_ds = hf_ds
        if wav_paths is None and hf_ds is None:
            raise ValueError(
                "Must provide one of either wav_paths or hf_ds, both were None"
            )
        if hf_ds is not None:
            if "audio" not in hf_ds.features:
                raise ValueError("'audio' feature not found in HF dataset")
            if not isinstance(hf_ds.features["audio"], Audio):
                raise ValueError("'audio' feature not of expected type")
        self.sampling_rate = sampling_rate
        self.clip_seconds = clip_seconds
        self.target_len = int(sampling_rate * clip_seconds)
        print("================StreamingAudioWaveforms================")
        print("Loading and transforming audio data.")
        print("=======================================================")

    def __getitem__(self, i: int) -> torch.Tensor:
        if self.wav_paths is not None:
            fpath = self.wav_paths[i]
            x, source_sr = torchaudio.load(str(fpath))  # (C, T)
        elif self.hf_ds is not None:
            sample = self.hf_ds[i]["audio"].get_all_samples()
            x = sample.data  # (C, T)
            source_sr = sample.sample_rate
        else:
            raise ValueError("Unknown how to get sample")

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

        # crop or pad to fixed length
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
        if self.wav_paths is not None:
            n = len(self.wav_paths)
        elif self.hf_ds is not None:
            n = len(self.hf_ds)
        else:
            raise ValueError("Unknown how to get shape")
        return (n, 1, self.target_len)
