import ast
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from wfdb import rdsamp

from ..defines import (
    PTBXL_CLIPPED_MEANS,
    PTBXL_CLIPPED_STDS,
    PTBXL_LEAD_ORDER,
    PTBXL_LOWERS,
    PTBXL_TARGETS,
    PTBXL_UPPERS,
    SPLIT_T,
    STANDARD_LEAD_ORDER,
)
from ._base_ecg_dataset import BaseECGDataset, load_cached_data, validate_label_subset

ptbxl_lead_order = [l.lower() for l in PTBXL_LEAD_ORDER]
standard_lead_order = [l.lower() for l in STANDARD_LEAD_ORDER]
assert all([c == s for c, s in zip(ptbxl_lead_order, standard_lead_order)])

VAL_FOLD = 9
TEST_FOLD = 10


def get_ptbxl_labels(
    df: pd.DataFrame,
    label_subset: list[str] | None = None,
) -> np.ndarray:
    targets = PTBXL_TARGETS
    if label_subset is not None:
        validate_label_subset(label_subset, PTBXL_TARGETS)
        targets = label_subset
    label_idx = {l: i for i, l in enumerate(targets)}
    temp = df["scp_codes"].apply(lambda x: ast.literal_eval(x))
    labels = np.zeros((len(df), len(label_idx)))
    for i, label_dict in enumerate(temp):
        for k, v in label_dict.items():
            if k in label_idx:
                labels[i, label_idx[k]] = 1
    return labels


class PtbxlECGDataset(BaseECGDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        _path = Path(dataset_path)
        df = pd.read_csv(_path / "ptbxl_database.csv", index_col="ecg_id")
        if split == "train":
            mask = ~df["strat_fold"].isin({VAL_FOLD, TEST_FOLD})
        elif split == "val":
            mask = df["strat_fold"] == VAL_FOLD
        elif split == "test":
            mask = df["strat_fold"] == TEST_FOLD
        else:
            raise ValueError(f"Unknown split: {split}")
        df = df[mask]

        self.patient_ids = torch.as_tensor(df["patient_id"].astype(int).to_numpy())
        self.ecg_ids = torch.as_tensor(df.index.to_numpy())
        self.labels = torch.as_tensor(
            get_ptbxl_labels(df, label_subset),
            dtype=torch.long,
        )

        def load_transform_data_fn() -> torch.Tensor:
            if sampling_rate <= 100:
                source_freq = 100
                data = [rdsamp(_path / f) for f in df["filename_lr"]]
            else:
                source_freq = 500
                data = [rdsamp(_path / f) for f in df["filename_hr"]]
            X = []
            for signal, meta in data:
                X.append(signal)
                lead_order = [l.lower() for l in meta["sig_name"]]
                assert all([c == l for c, l in zip(ptbxl_lead_order, lead_order)])
            # (N, 10 * source_freq, 12)
            X = np.array(X)

            # clip and normalize using stats derived over train set
            X = np.clip(X, PTBXL_LOWERS, PTBXL_UPPERS)
            X = (X - PTBXL_CLIPPED_MEANS) / PTBXL_CLIPPED_STDS

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
        assert self.labels is None or self.patient_ids.shape[0] == self.labels.shape[0]
