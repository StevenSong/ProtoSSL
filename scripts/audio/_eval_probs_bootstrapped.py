import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import train_test_split

from protossl.datasets import infer_dataset_class_from_path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--probs-npy", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--bootstrap-frac", type=float, default=0.5)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    dataset_path: str,
    probs_npy: str,
    output_path: str,
    n_bootstraps: int = 1000,
    bootstrap_frac: float = 0.5,
):
    ds_cls, label_names, is_audio = infer_dataset_class_from_path(dataset_path)
    if not is_audio:
        raise ValueError("This eval script is meant for audio related work")
    test_ds = ds_cls(
        dataset_path=dataset_path,
        split="test",
        sampling_rate=32000,
    )
    assert test_ds.labels is not None and label_names is not None

    y = test_ds.labels.numpy()  # (N, n_labels)
    y_prob: np.ndarray = np.load(probs_npy, allow_pickle=True)  # (N, n_labels)

    # ensure one-hot targets are multiclass
    assert y.ndim == 2, "Expected 2D matrix (N, num_classes)"
    assert (y.sum(axis=-1) == 1).all(), "Rows must sum to 1"
    assert (y >= 0).all() and (y <= 1).all(), "Values must be 0 or 1"

    os.makedirs(output_path, exist_ok=True)
    assert not os.path.exists(os.path.join(output_path, "metrics-bootstrapped.csv"))

    y_true = y.argmax(axis=-1)  # convert to multiclass
    y_pred = y_prob.argmax(axis=-1)

    bootstrap_n = int(len(y_true) * bootstrap_frac)
    bootstrapped_metrics = defaultdict(lambda: defaultdict(list))
    for b in range(n_bootstraps):
        bootstrap_y_true, _, bootstrap_y_pred, _ = train_test_split(
            y_true,
            y_pred,
            train_size=bootstrap_n,
            random_state=b,
            stratify=y_true,
        )
        acc = (bootstrap_y_true == bootstrap_y_pred).sum() / len(bootstrap_y_true)
        bootstrapped_metrics["Multiclass"]["Accuracy"].append(acc)

    metrics = defaultdict(dict)
    for label_name, label_metrics in bootstrapped_metrics.items():
        for metric_name, vs in label_metrics.items():
            a = np.asarray(vs)
            a_avg = np.mean(a)
            lo, hi = st.t.interval(0.95, len(a) - 1, loc=a_avg, scale=st.sem(a))
            metrics[label_name][metric_name] = a_avg
            metrics[label_name][f"{metric_name} 95% CI (lo)"] = lo
            metrics[label_name][f"{metric_name} 95% CI (hi)"] = hi

    metrics = pd.DataFrame.from_dict(metrics, orient="index")
    metrics.index.name = "Label"

    metrics_path = os.path.join(output_path, "metrics-bootstrapped.csv")
    metrics.to_csv(metrics_path)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset_path,
        probs_npy=args.probs_npy,
        output_path=args.output_path,
        n_bootstraps=args.n_bootstraps,
        bootstrap_frac=args.bootstrap_frac,
    )
