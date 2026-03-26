import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from pass_pclr.datasets import EchoNextECGDataset, infer_dataset_class_from_path
from pass_pclr.defines import ECHONEXT_COMPOSITE_TARGET


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--probs-npy", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--label-subset", nargs="+")
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    dataset_path: str,
    probs_npy: str,
    output_path: str,
    label_subset: list[str] | None = None,
):
    ds_cls, label_names = infer_dataset_class_from_path(dataset_path)
    test_ds = ds_cls(
        dataset_path=dataset_path,
        split="test",
        sampling_rate=100,
        label_subset=label_subset,
    )
    assert test_ds.labels is not None and label_names is not None

    composite_target = None
    if ds_cls == EchoNextECGDataset:
        composite_target = ECHONEXT_COMPOSITE_TARGET

    test_targets = test_ds.labels.numpy()
    target_probs = np.load(probs_npy, allow_pickle=True)

    os.makedirs(output_path, exist_ok=True)
    assert not os.path.exists(os.path.join(output_path, "metrics.csv"))

    multilabel_true = []
    multilabel_prob = []
    composite_true = None
    composite_prob = None
    metrics = defaultdict(dict)
    for i, target_col in enumerate(label_names):
        y_test = test_targets[:, i]
        y_prob = target_probs[:, i]

        if target_col != composite_target:
            metrics[target_col]["AUROC"] = roc_auc_score(y_test, y_prob)
            metrics[target_col]["AUPRC"] = average_precision_score(y_test, y_prob)
            multilabel_true.append(y_test)
            multilabel_prob.append(y_prob)
        else:
            assert composite_true is None, "Cannot have more than 1 composite"
            assert composite_prob is None, "Cannot have more than 1 composite"
            composite_true = y_test
            composite_prob = y_prob

    multilabel_true = np.asarray(multilabel_true).T
    multilabel_prob = np.asarray(multilabel_prob).T
    auroc = roc_auc_score(multilabel_true, multilabel_prob, average="macro")
    auprc = average_precision_score(multilabel_true, multilabel_prob, average="macro")
    metrics["Multilabel Averaged"]["AUROC"] = auroc
    metrics["Multilabel Averaged"]["AUPRC"] = auprc

    if composite_true is not None and composite_prob is not None:
        metrics[ECHONEXT_COMPOSITE_TARGET]["AUROC"] = roc_auc_score(
            composite_true, composite_prob
        )
        metrics[ECHONEXT_COMPOSITE_TARGET]["AUPRC"] = average_precision_score(
            composite_true, composite_prob
        )

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
        label_subset=args.label_subset,
    )
