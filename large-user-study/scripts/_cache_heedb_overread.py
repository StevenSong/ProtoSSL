import os

os.environ["HIGH_MEMORY"] = "1"

from protossl.datasets import HeedbECGDataset

DATASET_PATH = "/opt/gpudata/ecg/heedb"

# builds the overread mask and waveform caches up front, so the concurrently
# launched arm jobs only ever read them
for split in ["train", "val", "test"]:
    HeedbECGDataset(
        dataset_path=DATASET_PATH,
        sampling_rate=100,
        split=split,
        heedb_split_type="by-label",
        drop_blank_overread=True,
    )
