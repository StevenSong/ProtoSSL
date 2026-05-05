import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as st
from pqdm.processes import pqdm
from sklearn.metrics import average_precision_score, roc_auc_score

from protossl.datasets import EchoNextECGDataset, infer_dataset_class_from_path
from protossl.defines import ECHONEXT_COMPOSITE_TARGET


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--probs-npy", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--label-subset", nargs="+")
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--bootstrap-frac", type=float, default=0.5)
    parser.add_argument("--n-jobs", type=int, default=24)
    args = parser.parse_args()
    return args


def worker(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int,
    bootstrap_frac: float,
) -> dict:
    bootstrap_n = int(len(y_test) * bootstrap_frac)
    label_pos_frac = y_test.sum() / len(y_test)
    label_metrics = dict()
    pos_idxs = np.argwhere(y_test).squeeze(1)
    neg_idxs = np.argwhere(1 - y_test).squeeze(1)

    bootstrapped_metrics = defaultdict(list)
    for b in range(n_bootstraps):
        rng = np.random.default_rng(b)
        bootstrap_n_pos = int(bootstrap_n * label_pos_frac)
        bootstrap_n_pos = max(bootstrap_n_pos, 1)  # require at least 1 sample
        bootstrap_n_neg = bootstrap_n - bootstrap_n_pos

        bootstrap_pos_idxs = rng.choice(pos_idxs, bootstrap_n_pos, replace=True)
        bootstrap_neg_idxs = rng.choice(neg_idxs, bootstrap_n_neg, replace=True)

        bootstrap_idxs = np.concat([bootstrap_pos_idxs, bootstrap_neg_idxs])

        bootstrap_y_test = y_test[bootstrap_idxs]
        bootstrap_y_prob = y_prob[bootstrap_idxs]

        assert bootstrap_y_test.sum() > 0

        bootstrapped_metrics["AUROC"].append(
            roc_auc_score(bootstrap_y_test, bootstrap_y_prob)
        )
        bootstrapped_metrics["AUPRC"].append(
            average_precision_score(bootstrap_y_test, bootstrap_y_prob)
        )

    for k, vs in bootstrapped_metrics.items():
        a = np.asarray(vs)
        a_avg = np.mean(a)
        lo, hi = st.t.interval(0.95, len(a) - 1, loc=a_avg, scale=st.sem(a))
        label_metrics[k] = a_avg
        label_metrics[f"{k} 95% CI (lo)"] = lo
        label_metrics[f"{k} 95% CI (hi)"] = hi

    return label_metrics


def main(
    *,  # enforce kwargs
    dataset_path: str,
    probs_npy: str,
    output_path: str,
    label_subset: list[str] | None = None,
    n_bootstraps: int = 1000,
    bootstrap_frac: float = 0.5,
    n_jobs: int = 24,
):
    ds_cls, src_label_names, is_audio = infer_dataset_class_from_path(dataset_path)
    if is_audio:
        raise ValueError("This eval script is meant for ECG related work")
    test_ds = ds_cls(
        dataset_path=dataset_path,
        split="test",
        sampling_rate=100,
        label_subset=label_subset,
    )
    assert test_ds.labels is not None and src_label_names is not None
    if label_subset is not None:
        label_names = label_subset
    else:
        label_names = src_label_names

    composite_target = None
    if ds_cls == EchoNextECGDataset:
        composite_target = ECHONEXT_COMPOSITE_TARGET

    test_targets = test_ds.labels.numpy()
    target_probs = np.load(probs_npy, allow_pickle=True)

    kwargs = [
        {
            "y_test": test_targets[:, src_label_names.index(target_col)],
            "y_prob": target_probs[:, src_label_names.index(target_col)],
            "n_bootstraps": n_bootstraps,
            "bootstrap_frac": bootstrap_frac,
        }
        for target_col in label_names
    ]

    results = pqdm(kwargs, worker, argument_type="kwargs", n_jobs=n_jobs)  # type: ignore
    metrics = {label_name: results[l] for l, label_name in enumerate(label_names)}

    metrics = pd.DataFrame.from_dict(metrics, orient="index")
    metrics.index.name = "Label"

    _labels = [x for x in label_names if x != composite_target]
    metrics.loc["Multilabel Averaged"] = metrics.loc[_labels].mean()

    metrics_path = os.path.join(output_path, "metrics-bootstrapped.csv")
    metrics.to_csv(metrics_path)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset_path,
        probs_npy=args.probs_npy,
        output_path=args.output_path,
        label_subset=args.label_subset,
        n_bootstraps=args.n_bootstraps,
        bootstrap_frac=args.bootstrap_frac,
        n_jobs=args.n_jobs,
    )
