import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset

from ..defines import SPEECH_COMMANDS_V2_TARGETS, SPLIT_T
from ._base_dataset import BaseTSDataset
from .streaming_loaders import StreamingAudioWaveforms


class SpeechCommandsV2Dataset(BaseTSDataset):
    def __init__(
        self,
        *,
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        """
        We use the huggingface version of this dataset. At the time of writing,
        the default implementation of the dataset still uses a legacy script to
        load the data. This is no longer supported since datasets>=4.0.0, which
        we use. There is however a PR to convert to the standard format:
        https://huggingface.co/datasets/google/speech_commands/discussions/5

        We download this revision and store locally using the following command:
        ```
        hf download --type dataset google/speech_commands --revision 41990749 --local-dir /path/to/local/speech-commands-v2
        ```
        """
        if label_subset is not None:
            raise NotImplementedError(
                f"label_subset not yet supported for {type(self.__name__)}"
            )

        _split = split
        if split == "val":
            _split = "validation"
        hf_ds: HFDataset = load_dataset(dataset_path, name="v0.02", split=_split)  # type: ignore
        pids, pid_to_id = pd.factorize(np.asarray(hf_ds["speaker_id"]))

        self.source_ids = torch.as_tensor(pids, dtype=torch.long)
        self.sample_ids = torch.arange(len(hf_ds), dtype=torch.long)

        label_to_idx = {k: i for i, k in enumerate(SPEECH_COMMANDS_V2_TARGETS)}
        hf_idx_to_label = {i: k for i, k in enumerate(hf_ds.features["label"].names)}

        y = torch.zeros((len(hf_ds), len(SPEECH_COMMANDS_V2_TARGETS)), dtype=torch.long)
        for i, hf_idx in enumerate(hf_ds["label"]):
            label = hf_idx_to_label[hf_idx]
            if label not in label_to_idx:
                continue
            idx = label_to_idx[label]
            y[i, idx] = 1
        self.labels = y

        self.waveforms = StreamingAudioWaveforms(
            hf_ds=hf_ds,
            sampling_rate=sampling_rate,
            clip_seconds=1.0,
        )

        assert self.source_ids.shape[0] == self.waveforms.shape[0]
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]
        assert self.source_ids.shape[0] == self.labels.shape[0]
