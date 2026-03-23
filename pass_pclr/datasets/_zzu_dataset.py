from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from wfdb import rdsamp

from pass_pclr.datasets import BaseECGDataset, load_cached_data
from pass_pclr.defines import (
    SPLIT_T,
    ZZU_CLIPPED_MEANS,
    ZZU_CLIPPED_STDS,
    ZZU_LOWERS,
    ZZU_TARGETS,
    ZZU_UPPERS,
)

TRAIN_VAL_SPLIT_DATE = "2023"
VAL_TEST_SPLIT_DATE = "2023-08"


def get_zzu_dataframe(dataset_path: str) -> pd.DataFrame:
    _path = Path(dataset_path)
    df = pd.read_csv(_path / "AttributesDictionary.csv")
    df = (
        df[df["Lead"] == 12]
        .sort_values(["Patient_ID", "ECG_ID"])
        .reset_index(drop=True)
    )

    # derive labels
    # create a long table of idx --> icd code, where idx can appear multiple time
    dxs = df["ICD-10 code"].str.split(";").explode()
    # convert to dict of icd code --> list[idx]
    dx_idxs = {icd.strip("'"): list(group.index) for icd, group in dxs.groupby(dxs)}

    # convert to wide
    for label, icds in ZZU_TARGETS.items():
        df[label] = 0
        for icd in icds:
            df.loc[dx_idxs[icd], label] = 1

    # create intrinsic data splits based on time
    # require first ECG per patient in val/test
    assert (df.sort_values(["Patient_ID", "ECG_ID"]).index == df.index).all()
    first_ecg_mask = ~df.duplicated("Patient_ID", keep="first")

    train_mask = df["Acquisition_date"] < TRAIN_VAL_SPLIT_DATE
    val_mask = (
        df["Acquisition_date"].between(TRAIN_VAL_SPLIT_DATE, VAL_TEST_SPLIT_DATE)
        & first_ecg_mask
    )
    test_mask = (df["Acquisition_date"] > VAL_TEST_SPLIT_DATE) & first_ecg_mask

    # check that all coarse grained labels present in val/test
    assert (df.loc[val_mask, list(ZZU_TARGETS)].sum() > 0).all()
    assert (df.loc[test_mask, list(ZZU_TARGETS)].sum() > 0).all()

    df["split"] = "no_split"
    df.loc[train_mask, "split"] = "train"
    df.loc[val_mask, "split"] = "val"
    df.loc[test_mask, "split"] = "test"

    df["fpath"] = _path / "Child_ecg" / df["Filename"]

    return df


class ZzuECGDataset(BaseECGDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
    ):
        df = get_zzu_dataframe(dataset_path)
        df = df[df["split"] == split]

        self.patient_ids = torch.as_tensor(df["Patient_ID"].to_numpy())
        self.ecg_ids = torch.as_tensor(df["ECG_ID"].to_numpy())
        self.labels = torch.as_tensor(
            df[list(ZZU_TARGETS)].to_numpy(), dtype=torch.long
        )

        # ZZU pECG waveforms are variable length!
        # so we'll pad and center crop to get 10 second recordings
        def load_transform_data_fn() -> torch.Tensor:
            data = []
            source_freq = 500
            target_time = 10  # seconds

            # pad to even length (memory inefficient but simpler)
            max_timesteps = df["Sampling_point"].max()
            for i, f in enumerate(df["fpath"]):
                signal, meta = rdsamp(f)
                assert signal is not None
                assert not np.isnan(signal).any()

                n_timesteps, n_leads = signal.shape
                assert df.loc[i, "Sampling_point"] == n_timesteps
                assert n_leads == 12
                assert meta["fs"] == source_freq
                assert n_timesteps <= max_timesteps
                pad_total = max_timesteps - n_timesteps
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                signal = np.pad(
                    signal,
                    ((pad_before, pad_after), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                data.append(signal)
            X = np.array(data)  # (N, max_timesteps, 12)

            # center crop
            center_timestep = max_timesteps // 2
            target_timesteps = target_time * source_freq
            start_timestep = center_timestep - target_timesteps // 2
            end_timestep = start_timestep + target_timesteps
            X = X[:, start_timestep:end_timestep, :]

            # clip and normalize using stats derived over train set
            X = np.clip(X, ZZU_LOWERS, ZZU_UPPERS)
            X = (X - ZZU_CLIPPED_MEANS) / ZZU_CLIPPED_STDS

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
