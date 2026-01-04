import argparse
import os
import sys

sys.path.append("../external/bbj-lab-protoecgnet/src")

FREQ = 100  # Hz
SPLITS = ["train", "val", "test"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--echonext-data", required=True)
    args = parser.parse_args()
    os.environ["ECHONEXT_DATA"] = args.echonext_data

    # must import after setting env var, gets read on import
    from ecg_utils import DATASET_PATH, load_cached_echonext_data  # type: ignore

    assert DATASET_PATH == args.echonext_data

    for split in SPLITS:
        load_cached_echonext_data(
            sampling_rate=FREQ,
            split=split,
        )
