import argparse

from protossl.datasets import infer_dataset_class_from_path
from protossl.defines import SPLIT_T

FREQ = 100  # Hz
SPLITS: list[SPLIT_T] = ["train", "val", "test"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    args = parser.parse_args()

    ds_cls, _, is_audio = infer_dataset_class_from_path(args.dataset_path)
    if is_audio:
        raise ValueError(f"This utility primarily meant for ECG")

    for split in SPLITS:
        # caches data if it has not yet been loaded before
        ds_cls(
            dataset_path=args.dataset_path,
            split=split,
            sampling_rate=FREQ,
            label_subset=None,  # label subset does not matter for caching
        )
