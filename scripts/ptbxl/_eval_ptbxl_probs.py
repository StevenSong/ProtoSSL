import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from pass_pclr.datasets import get_ptbxl_labels
from pass_pclr.defines import PTBXL_TARGETS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ptbxl-data", required=True)
    parser.add_argument("--probs-npy", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    ptbxl_data: str,
    probs_npy: str,
    output_path: str,
):
    ptbxl_path = Path(ptbxl_data)

    df = pd.read_csv(ptbxl_path / "ptbxl_database.csv", index_col="ecg_id")

    target_probs = np.load(probs_npy, allow_pickle=True)

    os.makedirs(output_path, exist_ok=True)
    assert not os.path.exists(os.path.join(output_path, "metrics.csv"))

    test_df = df[df["strat_fold"] == 10]
    test_labels = get_ptbxl_labels(test_df)

    multilabel_true = []
    multilabel_prob = []
    metrics = defaultdict(dict)
    for i, target_col in enumerate(PTBXL_TARGETS):
        y_test = test_labels[:, i]
        y_prob = target_probs[:, i]

        metrics[target_col]["AUROC"] = roc_auc_score(y_test, y_prob)
        metrics[target_col]["AUPRC"] = average_precision_score(y_test, y_prob)
        multilabel_true.append(y_test)
        multilabel_prob.append(y_prob)

    multilabel_true = np.asarray(multilabel_true).T
    multilabel_prob = np.asarray(multilabel_prob).T
    auroc = roc_auc_score(multilabel_true, multilabel_prob, average="macro")
    auprc = average_precision_score(multilabel_true, multilabel_prob, average="macro")
    metrics["Multilabel Averaged"]["AUROC"] = auroc
    metrics["Multilabel Averaged"]["AUPRC"] = auprc

    metrics = pd.DataFrame.from_dict(metrics, orient="index")
    metrics.index.name = "Label"

    metrics_path = os.path.join(output_path, "metrics.csv")
    metrics.to_csv(metrics_path)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        ptbxl_data=args.ptbxl_data,
        probs_npy=args.probs_npy,
        output_path=args.output_path,
    )
