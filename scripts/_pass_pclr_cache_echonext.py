import argparse

from pass_pclr.datasets import infer_dataset_class_from_path
from pass_pclr.defines import SPLIT_T

FREQ = 100  # Hz
SPLITS: list[SPLIT_T] = ["train", "val", "test"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--echonext-data", required=True)
    args = parser.parse_args()

    ds_cls, _ = infer_dataset_class_from_path(args.echonext_data)

    for split in SPLITS:
        ds_cls(
            dataset_path=args.echonext_data,
            split=split,
            sampling_rate=FREQ,
        )
