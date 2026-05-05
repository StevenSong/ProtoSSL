from pathlib import Path

import pandas as pd
import torch
from datasets import IterableDataset, load_dataset

from ..defines import SPLIT_T
from ._base_dataset import BaseTSDataset
from ._utils import IndexedParquetDataset
from .streaming_loaders import StreamingAudioWaveforms


class AudioSetDataset(BaseTSDataset):
    def __init__(
        self,
        *,
        dataset_path: str,
        split: SPLIT_T,
        sampling_rate: int,
        label_subset: list[str] | None = None,
        use_cache: bool = True,
        augment_train: bool = True,
    ):
        """
        We use a huggingface version of this dataset provided by the community:
        https://huggingface.co/datasets/agkphysics/AudioSet

        We download and store this dataset and associated metadata using the following commands:
        ```
        hf download --type dataset agkphysics/AudioSet --local-dir /path/to/local/audioset
        wget -P /path/to/local/audioset https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
        wget -P /path/to/local/audioset https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv
        wget -P /path/to/local/audioset https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
        wget -P /path/to/local/audioset https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
        ```

        Lastly, because the hf dataset does not have all videos defined by the
        source metadata (due to various factors, such as video removal), we parse
        the list of valid IDs present in this version of audioset and provide these
        IDs in our repo. These should be placed alongside the rest of the dataset:
        ```
        cp /path/to/repo/data-preprocessing/hf_audioset_ids/* /path/to/local/audioset/.
        ```
        """

        if label_subset is not None:
            raise NotImplementedError(
                f"label_subset not yet supported for {type(self.__name__)}"
            )

        _path = Path(dataset_path)

        # Using dataset.load_dataset requires streaming for this large dataset.
        # We've implemented a custom sharded parquet dataset reader that allows
        # us to access data by index (the hf native streaming dataset does not),
        # however we still use load_dataset to grab the given Feature decoding.
        if split == "train":
            subdir = "unbal_train"
            csv_file = "unbalanced_train_segments.csv"
            id_file = "unbal_valid_ids.csv"
        elif split == "val":
            subdir = "bal_train"
            csv_file = "balanced_train_segments.csv"
            id_file = "bal_valid_ids.csv"
        elif split == "test":
            subdir = "eval"
            csv_file = "eval_segments.csv"
            id_file = "eval_valid_ids.csv"
        else:
            raise ValueError(f"Unknown split={split}")
        df = pd.read_csv(
            _path / csv_file,
            comment="#",
            header=None,
            skipinitialspace=True,
            names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
        )
        valid_ids = pd.read_csv(_path / id_file)["video_id"]
        assert valid_ids.isin(df["YTID"]).all()
        assert df["YTID"].is_unique and valid_ids.is_unique
        df = df.set_index("YTID").loc[valid_ids].reset_index()
        self._df = df

        hf_ds: IterableDataset = load_dataset(str(_path), split="train", streaming=True)  # type: ignore
        assert hf_ds.features is not None
        shard_files = [str(x) for x in (_path / "data" / subdir).glob("*.parquet")]
        parquet_indexer = IndexedParquetDataset(shard_files, hf_ds.features)
        assert len(parquet_indexer) == len(df)
        self._parquet_indexer = parquet_indexer

        class_df = pd.read_csv(_path / "class_labels_indices.csv")
        self.label_names = class_df["mid"].tolist()
        mid_to_idx = {mid: i for i, mid in enumerate(self.label_names)}
        ytid_codes, _ = pd.factorize(df["YTID"], sort=True)
        self.source_ids = torch.as_tensor(ytid_codes, dtype=torch.long)
        self.sample_ids = torch.arange(len(df), dtype=torch.long)

        self.labels = torch.zeros((len(df), len(self.label_names)), dtype=torch.long)
        for i, raw in enumerate(df["positive_labels"]):
            if pd.isna(raw):
                continue
            for mid in str(raw).split(","):
                mid = mid.strip()
                if mid in mid_to_idx:
                    self.labels[i, mid_to_idx[mid]] = 1

        self.waveforms = StreamingAudioWaveforms(
            parquet_indexer=parquet_indexer,
            sampling_rate=sampling_rate,
            clip_seconds=10.0,
            # for audioset, we augment train by default, regardless of contrastive or not
            do_augmentation=(split == "train") and augment_train,
            gain_prob=0.2,
            gain_db_min=-6.0,
            gain_db_max=6.0,
            noise_prob=0.2,
            noise_std_min=1e-4,
            noise_std_max=2e-3,
            use_cache=use_cache,
        )

        assert self.source_ids.shape[0] == self.waveforms.shape[0]
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]
        assert self.source_ids.shape[0] == self.labels.shape[0]
