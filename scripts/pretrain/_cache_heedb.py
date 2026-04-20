import os

os.environ["HIGH_MEMORY"] = "1"

from pass_pclr.datasets import HeedbECGDataset

DATASET_PATH = "/opt/gpudata/ecg/heedb"

for split in ["train", "val", "test"]:
    HeedbECGDataset(
        dataset_path=DATASET_PATH,
        sampling_rate=100,
        split=split,
    )
