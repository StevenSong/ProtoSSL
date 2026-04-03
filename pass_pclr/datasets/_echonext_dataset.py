from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

from ..defines import ECHONEXT_TARGETS, SPLIT_T
from ._base_ecg_dataset import BaseECGDataset, load_cached_data, validate_label_subset


class EchoNextECGDataset(BaseECGDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        targets = ECHONEXT_TARGETS
        if label_subset is not None:
            validate_label_subset(label_subset, list(ECHONEXT_TARGETS))
            targets = {label: ECHONEXT_TARGETS[label] for label in label_subset}
        mapping = {v: k for k, v in targets.items()}  # col --> name
        target_cols = list(mapping.values())

        _path = Path(dataset_path)
        df = pd.read_csv(_path / "EchoNext_metadata_100k.csv")
        df = df.rename(columns=mapping)
        split_mask = df["split"] == split
        id_df = df.loc[split_mask, ["patient_key", "ecg_key"]].reset_index(drop=True)
        label_df = df.loc[split_mask, target_cols].reset_index(drop=True)

        self.patient_ids = torch.as_tensor(id_df["patient_key"].to_numpy())
        self.ecg_ids = torch.as_tensor(id_df["ecg_key"].to_numpy())
        self.labels = torch.as_tensor(
            label_df.to_numpy(),
            dtype=torch.long,
        )  # (N, num_classes)

        def load_transform_data_fn() -> torch.Tensor:
            X = np.load(_path / f"EchoNext_{split}_waveforms.npy")  # (N, 1, 2500, 12)
            X = X.squeeze(1)  # (N, 2500, 12)

            # echonext comes prenormalized
            # https://github.com/PierreElias/IntroECG/blob/2361433c4cbdd29c01229f4e7b3216a38eff87b5/7-EchoNext%20Minimodel/preprocess.py#L89

            # downsample to target frequency
            if sampling_rate != 250:
                resample_frac = Fraction(
                    numerator=sampling_rate,
                    denominator=250,  # EchoNext preprocessed data is 250 Hz
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
