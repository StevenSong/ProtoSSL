from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

from ..defines import (
    CODE15_CLIPPED_MEANS,
    CODE15_CLIPPED_STDS,
    CODE15_LEAD_ORDER,
    CODE15_LOWERS,
    CODE15_TARGETS,
    CODE15_UPPERS,
    SPLIT_T,
    STANDARD_LEAD_ORDER,
)
from ._base_ecg_dataset import BaseECGDataset, load_cached_data, validate_label_subset

code15_lead_order = [l.lower() for l in CODE15_LEAD_ORDER]
standard_lead_order = [l.lower() for l in STANDARD_LEAD_ORDER]
assert all([c == s for c, s in zip(code15_lead_order, standard_lead_order)])


class Code15ECGDataset(BaseECGDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        targets = CODE15_TARGETS
        if label_subset is not None:
            validate_label_subset(label_subset, CODE15_TARGETS)
            targets = label_subset
        _path = Path(dataset_path)
        df = pd.read_csv(_path / "labels.csv")
        df = df[df["split"] == split]

        self.patient_ids = torch.as_tensor(df["patient_id"].to_numpy())
        self.ecg_ids = torch.as_tensor(df["exam_id"].to_numpy())
        self.labels = torch.as_tensor(df[targets].to_numpy(), dtype=torch.long)

        def load_transform_data_fn() -> torch.Tensor:
            # based on the zenodo and paper:
            # https://zenodo.org/records/4916206
            # https://www.nature.com/articles/s41467-020-15432-4
            # data is 400 hz for 7 or (maybe between?) 10 seconds and padded,
            # but it's unclear why this results in 4096 timesteps.
            # if 10 seconds had been padded, shouldn't this result in everything
            # having leading/trailing 0s? but this isn't observed...
            # going to assume it is 400 Hz and just center crop
            source_freq = 400

            # gather waveforms from all shards
            all_waveforms = []
            shard_lens = []
            n_shards = 18
            for i in range(n_shards):
                with h5py.File(_path / f"exams_part{i}.hdf5", "r") as h5:
                    data: np.ndarray = h5["tracings"][:]  # type: ignore - (N, 4096, 12)
                shard_lens.append(data.shape[0])

                # do center crop to 10 seconds
                assert data.shape[1:] == (4096, 12)
                data = data[:, 48:-48, :]  # (N, 4000, 12)
                assert data.shape[1:] == (4000, 12)

                all_waveforms.append(data)
            all_waveforms = np.concatenate(all_waveforms)

            # convert shard indices into global index
            shard_starts = np.cumsum(shard_lens) - shard_lens[0]
            shard_to_offset = {i: s for i, s in enumerate(shard_starts)}
            global_idx = df["shard_num"].replace(shard_to_offset) + df["shard_idx"]
            assert global_idx.is_unique

            X = all_waveforms[global_idx.to_numpy()]  # (N, 10 * source_freq, 12)

            # clip and normalize using stats derived over train set
            X = np.clip(X, CODE15_LOWERS, CODE15_UPPERS)
            X = (X - CODE15_CLIPPED_MEANS) / CODE15_CLIPPED_STDS

            # downsample to target frequency
            if sampling_rate != source_freq:
                resample_frac = Fraction(
                    numerator=sampling_rate,
                    denominator=source_freq,
                ).limit_denominator(100)
                X = resample_poly(
                    X,
                    up=resample_frac.numerator,
                    down=resample_frac.denominator,
                    axis=1,
                )  # (N, 10 * sampling_rate, 12)
            X = torch.as_tensor(X).mT  # (N, 12, 10 * sampling_rate)
            return X

        self.waveforms = load_cached_data(
            load_transform_data_fn=load_transform_data_fn,
            dataset_path=dataset_path,
            split=split,
            sampling_rate=sampling_rate,
        )

        assert self.patient_ids.shape[0] == self.waveforms.shape[0]
        assert self.patient_ids.shape[0] == self.ecg_ids.shape[0]
        assert self.patient_ids.shape[0] == self.labels.shape[0]
