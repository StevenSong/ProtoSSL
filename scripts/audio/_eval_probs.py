import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd

from protossl.datasets import SpeechCommandsV2Dataset, infer_dataset_class_from_path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--probs-npy", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    dataset_path: str,
    probs_npy: str,
    output_path: str,
):
    ds_cls, label_names, is_audio = infer_dataset_class_from_path(dataset_path)
    if not is_audio:
        raise ValueError("This eval script is meant for audio related work")
    ignore_labels = None
    if ds_cls == SpeechCommandsV2Dataset:
        ignore_labels = ["_silence_"]  # only compute accuracy over 35 classes
    test_ds = ds_cls(
        dataset_path=dataset_path,
        split="test",
        sampling_rate=32000,
    )
    assert test_ds.labels is not None and label_names is not None

    # ensure one-hot targets are multiclass
    y = test_ds.labels.numpy()  # (N, n_labels)
    assert y.ndim == 2, "Expected 2D matrix (N, num_classes)"
    assert (y.sum(axis=-1) == 1).all(), "Rows must sum to 1"
    assert (y >= 0).all() and (y <= 1).all(), "Values must be 0 or 1"

    # mask out ignored labels, we're just computing accuracy so argmax will remain the same
    y_prob: np.ndarray = np.load(probs_npy, allow_pickle=True)  # (N, n_labels)
    if ignore_labels is not None:
        for ignore in ignore_labels:
            assert (
                ignore in label_names
            ), f"{ignore} is not a valid label to ignore (not in full label list)"
            idx = label_names.index(ignore)
            if y_prob[:, idx].sum() > 0:
                print(f"====================eval_probs=====================")
                print(f"Ignored label ({ignore}) only applies to predicted ")
                print(f"probabilities but has non-zero entries in the ground truth!")
                print(f"By ignoring this label, the model will always be wrong")
                print(f"and penalized for instances of this ignored label!")
                print(f"===================================================")
            y_prob[:, idx] = -np.inf

    os.makedirs(output_path, exist_ok=True)
    assert not os.path.exists(os.path.join(output_path, "metrics.csv"))

    metrics = defaultdict(dict)
    y_true = y.argmax(axis=-1)  # convert to multiclass
    y_pred = y_prob.argmax(axis=-1)

    acc = (y_true == y_pred).sum() / len(y_true)
    metrics["Multiclass"]["Accuracy"] = acc

    metrics = pd.DataFrame.from_dict(metrics, orient="index")
    metrics.index.name = "Label"

    metrics_path = os.path.join(output_path, "metrics.csv")
    metrics.to_csv(metrics_path)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset_path,
        probs_npy=args.probs_npy,
        output_path=args.output_path,
    )
