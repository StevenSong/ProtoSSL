import argparse
import os
import sys

sys.path.append("../external/bbj-lab-protoecgnet/src")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl-data", required=True)
    args = parser.parse_args()
    os.environ["DATASET_PATH"] = args.ptbxl_data

    # must import after setting env var, gets read on import
    from ecg_utils import DATASET_PATH, get_ptbxl_dataloaders  # type: ignore

    assert DATASET_PATH == args.ptbxl_data

    get_ptbxl_dataloaders(
        batch_size=32,
        mode="1D",
        sampling_rate=100,
        label_set="1",
        work_num=4,
        return_sample_ids=False,
        custom_groups=True,
        standardize=True,
        remove_baseline=True,
    )
