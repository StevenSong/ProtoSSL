from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import resample_poly

from ._streaming_base import StreamingWaveformsBase


class StreamingAudioWaveforms(StreamingWaveformsBase):
    def __init__(
        self,
        *,  # enforce kwargs
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
