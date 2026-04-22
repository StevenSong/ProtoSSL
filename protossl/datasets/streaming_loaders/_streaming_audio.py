from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from cachetools import TTLCache
from datasets import Dataset as HFDataset
from datasets.features import Audio
from scipy.signal import resample_poly

from .._utils import IndexedParquetDataset
from ._streaming_base import StreamingWaveformsBase


class StreamingAudioWaveforms(StreamingWaveformsBase):
    def __init__(
        self,
        *,  # enforce kwargs
        wav_paths: list[Path] | None = None,
        hf_ds: HFDataset | None = None,
        parquet_indexer: IndexedParquetDataset | None = None,
        sampling_rate: int,
        clip_seconds: float = 10.0,
        use_cache: bool = True,
        do_augmentation: bool = False,
        # below only matters if do_augmentation - user must correctly only use augmentation for training subset
        gain_prob: float = 0.2,
        gain_db_min: float = -6.0,
        gain_db_max: float = 6.0,
        noise_prob: float = 0.2,
        noise_std_min: float = 1e-4,
        noise_std_max: float = 2e-3,
    ):
        self.wav_paths = wav_paths
        self.hf_ds = hf_ds
        self.parquet_indexer = parquet_indexer
        if wav_paths is None and hf_ds is None and parquet_indexer is None:
            raise ValueError(
                "Must provide one of wav_paths, hf_ds, or parquet_indexer, all were None"
            )
        if sum([s is not None for s in [wav_paths, hf_ds, parquet_indexer]]) > 1:
            # mutually exclusive data read options
            raise ValueError(
                "Can only provide one of wav_paths, hf_ds, or parquet_indexer, got multiple"
            )
        if hf_ds is not None:
            if "audio" not in hf_ds.features:
                raise ValueError("'audio' feature not found in HF dataset")
            if not isinstance(hf_ds.features["audio"], Audio):
                raise ValueError("'audio' feature not of expected type")
        self.sampling_rate = sampling_rate
        self.clip_seconds = clip_seconds
        self.target_len = int(sampling_rate * clip_seconds)

        self.do_augmentation = do_augmentation
        self.gain_prob = gain_prob
        self.gain_db_min = gain_db_min
        self.gain_db_max = gain_db_max
        self.noise_prob = noise_prob
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

        self.use_cache = use_cache
        self._cache = None  # initialized per-worker, see below

        # just some descriptive logging
        if self.wav_paths is not None:
            desc = "per-sample .wav files"
        elif self.hf_ds is not None:
            desc = "in-memory huggingface dataset"
        elif self.parquet_indexer is not None:
            desc = "custom parquet indexing of streaming huggingface dataset"
        else:
            raise ValueError("Unknown streaming method")
        print(f"================StreamingAudioWaveforms================")
        print(f"Loading and transforming audio data using {desc}.")
        if use_cache:
            print("Using in-memory cache to speed up fast, repeated reads.")
        print(f"=======================================================")

    @property
    def shape(self) -> tuple[int, ...]:
        if self.wav_paths is not None:
            n = len(self.wav_paths)
        elif self.hf_ds is not None:
            n = len(self.hf_ds)
        elif self.parquet_indexer is not None:
            n = len(self.parquet_indexer)
        else:
            raise ValueError("Unknown how to get shape")
        return (n, 1, self.target_len)

    def _get_cache(self):
        if self.use_cache:
            if self._cache is None:
                # Fast timeout (10 seconds) to improve reading the same sample
                # multiple times (e.g. for contrastive paired data augmentation).
                # Only create cache on first data read after dataset workers have
                # been created to prevent potential issues with threading.
                self._cache = TTLCache(maxsize=256, ttl=10)
            return self._cache
        else:
            # fake the cache (use temp dict that will get cleaned up)
            return dict()

    def _load_waveform(self, i: int) -> torch.Tensor:
        cache = self._get_cache()
        if i in cache:
            x, source_sr = cache[i]
        else:
            if self.wav_paths is not None:
                fpath = self.wav_paths[i]
                x, source_sr = torchaudio.load(str(fpath))  # (C, T)
            elif self.hf_ds is not None:
                sample = self.hf_ds[i]["audio"].get_all_samples()
                x = sample.data  # (C, T)
                source_sr = sample.sample_rate
            elif self.parquet_indexer is not None:
                sample = self.parquet_indexer[i]
                try:
                    waveform = sample["audio"].get_all_samples()
                    x = waveform.data  # (C, T)
                    source_sr = waveform.sample_rate
                except RuntimeError as e:
                    # exceedingly rare corrupted data, fill with zero signal
                    print(e)
                    x = torch.zeros([1, self.target_len], dtype=torch.float32)
                    source_sr = self.sampling_rate
            else:
                raise ValueError("Unknown how to load waveform")
        cache[i] = (x, source_sr)  # refresh or place in cache

        # convert to mono
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)  # (1, T)

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

        return x

    def _crop_or_pad(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        cur_len = x.shape[1]

        if cur_len > target_len:
            if self.do_augmentation:
                max_start = cur_len - target_len
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start = 0
            x = x[:, start : start + target_len]
        elif cur_len < target_len:
            pad_total = target_len - cur_len
            if self.do_augmentation:
                left: int = torch.randint(0, pad_total + 1, (1,)).item()  # type: ignore
            else:
                left = 0
            right = pad_total - left
            x = F.pad(x, (left, right))

        return x

    def _maybe_augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.do_augmentation:
            return x

        if torch.rand(1).item() < self.gain_prob:
            gain_db = torch.empty(1).uniform_(self.gain_db_min, self.gain_db_max).item()
            gain = 10 ** (gain_db / 20.0)
            x = x * gain

        if torch.rand(1).item() < self.noise_prob:
            noise_std = (
                torch.empty(1).uniform_(self.noise_std_min, self.noise_std_max).item()
            )
            x = x + torch.randn_like(x) * noise_std

        return x

    def sample_view(self, i: int, clip_seconds: float | None = None) -> torch.Tensor:
        x = self._load_waveform(i)

        target_len = (
            self.target_len
            if clip_seconds is None
            else int(self.sampling_rate * clip_seconds)
        )

        x = self._crop_or_pad(x, target_len=target_len)
        x = self._maybe_augment(x)

        # simple peak normalization
        peak = x.abs().max()
        if peak > 0:
            x = x / peak

        return x  # (1, T)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.sample_view(i)
