import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from protossl.datasets import EchoNextECGDataset, infer_dataset_class_from_path
from protossl.defines import ECHONEXT_COMPOSITE_TARGET


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--probs-npy", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--label-subset", nargs="+")
    args = parser.parse_args()
    return args


def is_audioset_path(dataset_path: str) -> bool:
    dataset_path_lower = dataset_path.lower()
    audioset_indicators = ["audioset", "audio-set", "audio_set"]
    return any(x in dataset_path_lower for x in audioset_indicators)


def get_eval_sampling_rate(dataset_path: str) -> int:
    if is_audioset_path(dataset_path):
        return 32000
    return 100


def safe_auc_to_dprime(auc: float) -> float:
    # avoid inf at exactly 0 or 1
    auc = float(np.clip(auc, 1e-7, 1 - 1e-7))
    return float(np.sqrt(2.0) * norm.ppf(auc))


def has_both_classes(y_true: np.ndarray) -> bool:
    y_true = np.asarray(y_true)
    return np.unique(y_true).size >= 2


def evaluate_audioset(
    *,
    test_targets: np.ndarray,
    target_probs: np.ndarray,
    label_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = defaultdict(dict)
    per_class_rows = []

    ap_vals = []
    auc_vals = []
    dprime_vals = []

    for i, target_col in enumerate(label_names):
        y_test = test_targets[:, i]
        y_prob = target_probs[:, i]

        prevalence = float(np.mean(y_test))

        row = {
            "Label": target_col,
            "Prevalence": prevalence,
            "NumPos": int(np.sum(y_test)),
            "NumNeg": int(len(y_test) - np.sum(y_test)),
        }

        # AP is usually still meaningful when positives exist.
        # If a class has zero positives, skip AP.
        if np.sum(y_test) > 0:
            ap = average_precision_score(y_test, y_prob)
            row["AP"] = ap
            ap_vals.append(ap)
        else:
            row["AP"] = np.nan

        # AUC / d-prime require both classes present
        if has_both_classes(y_test):
            auc = roc_auc_score(y_test, y_prob)
            dprime = safe_auc_to_dprime(auc)
            row["AUC"] = auc
            row["d_prime"] = dprime
            auc_vals.append(auc)
            dprime_vals.append(dprime)
        else:
            row["AUC"] = np.nan
            row["d_prime"] = np.nan

        per_class_rows.append(row)

    metrics["Multilabel Averaged"]["mAP"] = (
        float(np.mean(ap_vals)) if len(ap_vals) > 0 else np.nan
    )
    metrics["Multilabel Averaged"]["mAUC"] = (
        float(np.mean(auc_vals)) if len(auc_vals) > 0 else np.nan
    )
    metrics["Multilabel Averaged"]["d_prime"] = (
        float(np.mean(dprime_vals)) if len(dprime_vals) > 0 else np.nan
    )
    metrics["Multilabel Averaged"]["n_valid_ap_classes"] = len(ap_vals)
    metrics["Multilabel Averaged"]["n_valid_auc_classes"] = len(auc_vals)

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.index.name = "Label"

    per_class_df = pd.DataFrame(per_class_rows)
    return metrics_df, per_class_df


def evaluate_ecg(
    *,
    ds_cls,
    label_names: list[str],
    src_label_names: list[str],
    test_targets: np.ndarray,
    target_probs: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    composite_target = None
    if ds_cls == EchoNextECGDataset:
        composite_target = ECHONEXT_COMPOSITE_TARGET

    multilabel_true = []
    multilabel_prob = []
    composite_true = None
    composite_prob = None
    metrics = defaultdict(dict)
    for target_col in label_names:
        i = src_label_names.index(target_col)
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

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.index.name = "Label"
    return metrics_df, None


def main(
    *,  # enforce kwargs
    dataset_path: str,
    probs_npy: str,
    output_path: str,
    label_subset: list[str] | None = None,
):
    ds_cls, src_label_names = infer_dataset_class_from_path(dataset_path)
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

    test_targets = test_ds.labels.numpy()
    target_probs = np.load(probs_npy, allow_pickle=True)

    os.makedirs(output_path, exist_ok=True)

    metrics_path = os.path.join(output_path, "metrics.csv")
    per_class_path = os.path.join(output_path, "per_class_metrics.csv")

    assert not os.path.exists(metrics_path), f"{metrics_path} already exists"

    if is_audioset_path(dataset_path):
        metrics_df, per_class_df = evaluate_audioset(
            test_targets=test_targets,
            target_probs=target_probs,
            label_names=label_names,
        )
    else:
        metrics_df, per_class_df = evaluate_ecg(
            ds_cls=ds_cls,
            label_names=label_names,
            src_label_names=src_label_names,
            test_targets=test_targets,
            target_probs=target_probs,
        )

    metrics_df.to_csv(metrics_path)
    print(f"Saved metrics to {metrics_path}")

    if per_class_df is not None:
        per_class_df.to_csv(per_class_path, index=False)
        print(f"Saved per-class metrics to {per_class_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset_path,
        probs_npy=args.probs_npy,
        output_path=args.output_path,
        label_subset=args.label_subset,
    )
