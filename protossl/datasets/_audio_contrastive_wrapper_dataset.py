import torch
from torch.utils.data import Dataset

from ..defines import CONTRASTIVE_T
from ._audioset_dataset import AudioSetDataset


class AudioContrastiveWrapperDataset(Dataset):
    def __init__(
        self,
        dataset: AudioSetDataset,
        *,
        pair_mode: CONTRASTIVE_T = "cola",
        cola_view_seconds: float | None = None,
    ):
        self.ds = dataset
        self.pair_mode = pair_mode
        if pair_mode in {"cola", "cola+clar"} and cola_view_seconds is None:
            raise ValueError("Must set cola_view_seconds if pair_mode uses COLA")
        self.cola_view_seconds = cola_view_seconds

        if pair_mode not in {"cola", "clar", "cola+clar"}:
            raise ValueError(f"Unknown audio contrastive pair_mode={pair_mode}. ")

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        base1 = self.ds[i]

        ret = {
            "patient_id": base1["patient_id"],
        }

        if self.pair_mode in {"cola", "cola+clar"}:
            ret["x1"] = self.ds.sample_view(i, clip_seconds=self.cola_view_seconds)
            ret["x2"] = self.ds.sample_view(i, clip_seconds=self.cola_view_seconds)

        if self.pair_mode in {"clar", "cola+clar"}:
            base2 = self.ds[i]
            ret["x1_clar"] = base1["waveform"]
            ret["x2_clar"] = base2["waveform"]

            if self.pair_mode == "clar":
                ret["x1"] = ret["x1_clar"]
                ret["x2"] = ret["x2_clar"]

        return ret

    def __len__(self) -> int:
        return len(self.ds)
