import os

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset

from ..defines import IEMOCAP_TARGETS, SPLIT_T
from ._base_dataset import BaseTSDataset
from .streaming_loaders import StreamingAudioWaveforms

N_FOLDS = 5
DEFAULT_IEMOCAP_TEST_SPLIT = 0
ENV_VAR_NAME = "IEMOCAP_TEST_FOLD"


class IemocapDataset(BaseTSDataset):
    def __init__(
        self,
        *,
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
    ):
        """
        Data downloaded from: https://huggingface.co/datasets/mteb/iemocap
        """
        if label_subset is not None:
            raise NotImplementedError(
                f"label_subset not yet supported for {type(self.__name__)}"
            )

        test_fold = int(os.environ.get(ENV_VAR_NAME, DEFAULT_IEMOCAP_TEST_SPLIT))
        val_fold = (test_fold + 1) % N_FOLDS
        train_folds = [i for i in range(N_FOLDS) if i not in {test_fold, val_fold}]
        print(f"======================IEMOCAP======================")
        print(
            f"Using fold {test_fold} for test split and fold {val_fold} for val split."
        )
        print(
            f"To change this behavior, set the env var '{ENV_VAR_NAME}' to a value [0-{N_FOLDS-1}]."
        )
        print(f"NOTE: we reindex the source dataset folds to start from 0.")
        print(f"Fold used for val is automatically the fold after the one for test.")
        print(f"===================================================")

        if split == "train":
            use_folds = train_folds
        elif split == "val":
            use_folds = [val_fold]
        elif split == "test":
            use_folds = [test_fold]
        else:
            raise ValueError(f"Unknown fold {split}")

        # this version of the huggingface dataset puts all data in train
        hf_ds: HFDataset = load_dataset(dataset_path, split="train")  # type: ignore
        hf_ds = hf_ds.filter(
            lambda x: x in IEMOCAP_TARGETS,
            input_columns="major_emotion",
            load_from_cache_file=False,
        )
        hf_ds = hf_ds.filter(
            # index fold/session to start at 0
            lambda x: int(x[4:5]) - 1 in use_folds,
            input_columns="file",
            load_from_cache_file=False,
        )

        # TODO can consider making IDs more meaningful?
        self.source_ids = torch.arange(len(hf_ds), dtype=torch.long)
        self.sample_ids = torch.arange(len(hf_ds), dtype=torch.long)

        label_to_idx = {k: i for i, k in enumerate(IEMOCAP_TARGETS)}
        self.labels = torch.zeros((len(hf_ds), len(label_to_idx)), dtype=torch.long)
        for i, l in enumerate(hf_ds["major_emotion"]):
            idx = label_to_idx[l]
            self.labels[i, idx] = 1

        self.waveforms = StreamingAudioWaveforms(
            hf_ds=hf_ds,
            sampling_rate=sampling_rate,
            clip_seconds=4.5,  # average length of IEMOCAP segments
        )

        assert self.source_ids.shape[0] == self.waveforms.shape[0]
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]
        assert self.source_ids.shape[0] == self.labels.shape[0]
