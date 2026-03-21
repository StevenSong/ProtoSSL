import torch
from torch.utils.data import Dataset

from ._audioset_dataset import AudioSetDataset


class AudioContrastiveWrapperDataset(Dataset):
    def __init__(
        self,
        dataset: AudioSetDataset,
        *,
        pair_mode: str = "cola",
        cola_view_seconds: float | None = None,
    ):
        self.ds = dataset
        self.pair_mode = pair_mode
        self.cola_view_seconds = cola_view_seconds

        valid_modes = {"cola", "clar", "cola+clar"}
        if pair_mode not in valid_modes:
            raise ValueError(
                f"Unknown audio contrastive pair_mode={pair_mode}. "
                f"Expected one of {sorted(valid_modes)}"
            )

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        base = self.ds[i]
        x_full_1 = base["x"]

        ret = {
            "patient_id": base["patient_id"],
        }

        # COLA: two independently sampled temporal crops/views from the same clip
        if self.pair_mode in {"cola", "cola+clar"}:
            x1 = self.ds.sample_view(i, clip_seconds=self.cola_view_seconds)
            x2 = self.ds.sample_view(i, clip_seconds=self.cola_view_seconds)
            ret["x1"] = x1
            ret["x2"] = x2

        # CLAR: same full clip, two independent augmentations
        if self.pair_mode in {"clar", "cola+clar"}:
            x_full_2 = self.ds[i]["x"]

            ret["x1_clar"] = x_full_1
            ret["x2_clar"] = x_full_2

            if self.pair_mode == "clar":
                ret["x1"] = x_full_1
                ret["x2"] = x_full_2

        return ret

    def __len__(self) -> int:
        return len(self.ds)