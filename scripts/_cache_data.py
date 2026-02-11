import argparse

from pass_pclr.datasets import infer_dataset_class_from_path
from pass_pclr.defines import SPLIT_T

FREQ = 100  # Hz
SPLITS: list[SPLIT_T] = ["train", "val", "test"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    args = parser.parse_args()

    ds_cls, _ = infer_dataset_class_from_path(args.dataset_path)

    for split in SPLITS:
        # caches data if it has not yet been loaded before
        ds_cls(
            dataset_path=args.dataset_path,
            split=split,
            sampling_rate=FREQ,
        )
