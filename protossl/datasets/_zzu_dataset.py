from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from wfdb import rdsamp

from ..defines import (
    SPLIT_T,
    STANDARD_LEAD_ORDER,
    ZZU_CLIPPED_MEANS,
    ZZU_CLIPPED_STDS,
    ZZU_LEAD_ORDER,
    ZZU_LOWERS,
    ZZU_TARGETS,
    ZZU_UPPERS,
)
from ._base_ecg_dataset import BaseECGDataset, load_cached_data

zzu_lead_order = [l.lower() for l in ZZU_LEAD_ORDER]
standard_lead_order = [l.lower() for l in STANDARD_LEAD_ORDER]
assert all([c == s for c, s in zip(zzu_lead_order, standard_lead_order)])


def get_zzu_dataframe(dataset_path: str) -> pd.DataFrame:
    _path = Path(dataset_path)
    df = pd.read_csv(_path / "labels.csv")
    df["fpath"] = _path / "Child_ecg" / df["Filename"]

    # convert string IDs to numerical IDs:
    # all patient IDs are 6 digits, all ECG IDs are patient ID + 2 digits
    df["sample_id"] = df["sample_id"].str.replace("[^0-9]", "", regex=True)
    df["sample_id"] = ("1" + df["sample_id"]).astype(int)
    df["source_id"] = df["source_id"].str.replace("[^0-9]", "", regex=True)
    df["source_id"] = ("1" + df["source_id"]).astype(int)

    for label, sublabels in ZZU_TARGETS.items():
        df[label] = df[sublabels].max(axis=1)  # make coarse labels

    return df


class ZzuECGDataset(BaseECGDataset):
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        if label_subset is not None:
            raise NotImplementedError(
                f"label_subset not yet supported for {type(self.__name__)}"
            )
        df = get_zzu_dataframe(dataset_path)
        df = df[df["split"] == split]

        self.source_ids = torch.as_tensor(df["source_id"].to_numpy())
        self.sample_ids = torch.as_tensor(df["sample_id"].to_numpy())
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
            for i, f in df["fpath"].items():
                signal, meta = rdsamp(f)
                assert signal is not None
                assert not np.isnan(signal).any()
                lead_order = [l.lower() for l in meta["sig_name"]]
                assert all([c == l for c, l in zip(zzu_lead_order, lead_order)])

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
                    constant_values=np.nan,
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

            # fill nans after norm
            X[np.isnan(X)] = 0

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

        assert self.source_ids.shape[0] == self.waveforms.shape[0]
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]
        assert self.source_ids.shape[0] == self.labels.shape[0]
