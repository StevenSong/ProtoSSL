import torch
from torchaudio.datasets import VoxCeleb1Identification as TorchAudioVoxCeleb1Id

from ..defines import SPLIT_T, VOXCELEB1_ID_TARGETS
from ._base_dataset import BaseTSDataset
from ._utils import TypedDataset
from .streaming_loaders import StreamingAudioWaveforms


class VoxCeleb1IdDataset(BaseTSDataset):
    def __init__(
        self,
        *,
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        """
        The homepage for the dataset no longer publicly lists the request form
        to access and download the dataset. However, it seems the request form
        is still live and can be accessed here: https://cn01.mmai.io/keyreq/voxceleb
        (mmai.io is the lab page of one of the authors, JS Chung, now at KAIST)

        Dataset homepage: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
        Huggingface mirror: https://huggingface.co/datasets/ProgramComputer/voxceleb
        Based on the hashes provided by the automated data access form, the zips
        hosted on huggingface match and are correct.
        """
        if label_subset is not None:
            raise NotImplementedError(
                f"label_subset not yet supported for {type(self.__name__)}"
            )

        ds = TorchAudioVoxCeleb1Id(
            dataset_path,
            subset="dev" if split == "val" else split,
        )
        speakers = []
        for i in range(len(ds)):
            fpath, sr, speaker_id, fid = ds.get_metadata(i)
            speakers.append(speaker_id)

        self.source_ids = torch.as_tensor(speakers, dtype=torch.long)
        self.sample_ids = torch.arange(len(ds), dtype=torch.long)

        self.labels = torch.zeros(
            (len(ds), len(VOXCELEB1_ID_TARGETS)), dtype=torch.long
        )
        for i, speaker_id in enumerate(speakers):
            # speaker IDs start at 1, so need to sub 1
            self.labels[i, speaker_id - 1] = 1

        self.ds = ds

        def get_samp(i: int) -> tuple[torch.Tensor, int]:
            x, sr, speaker_id, fid = self.ds[i]
            return x, sr

        self.waveforms = StreamingAudioWaveforms(
            typed_ds=TypedDataset(get_samp=get_samp, n=len(ds)),
            sampling_rate=sampling_rate,
            clip_seconds=3.0,
        )

        assert self.source_ids.shape[0] == self.waveforms.shape[0]
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]
        assert self.source_ids.shape[0] == self.labels.shape[0]
