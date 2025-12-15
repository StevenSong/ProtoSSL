import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Literal, get_args
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

simplefilter("ignore", category=ConvergenceWarning)

TARGET_OPTS = Literal["multitask", "composite"]
MULTITASK_TARGET_NAMES = {
    "lvef_lte_45_flag": "LVEF Lo",
    "lvwt_gte_13_flag": "LVWT Hi",
    "aortic_stenosis_moderate_or_greater_flag": "AS",
    "aortic_regurgitation_moderate_or_greater_flag": "AR",
    "mitral_regurgitation_moderate_or_greater_flag": "MR",
    "tricuspid_regurgitation_moderate_or_greater_flag": "TR",
    "pulmonary_regurgitation_moderate_or_greater_flag": "PR",
    "rv_systolic_dysfunction_moderate_or_greater_flag": "RVD",
    "pericardial_effusion_moderate_large_flag": "PEff",
    "pasp_gte_45_flag": "PASP Hi",
    "tr_max_gte_32_flag": "TRV Hi",
}
COMPOSITE_TARGET_NAMES = {
    "shd_moderate_or_greater_flag": "SHD",  # composite binary label
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--echonext-data", required=True)
    parser.add_argument("--target-type", choices=get_args(TARGET_OPTS), required=True)
    parser.add_argument("--balance-class-weight", action="store_true")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    echonext_data: str,
    target_type: TARGET_OPTS,
    balance_class_weight: bool,
    output_path: str,
):
    echonext_path = Path(echonext_data)
    df = pd.read_csv(echonext_path / "EchoNext_metadata_100k.csv")
    X_test = np.load(echonext_path / "EchoNext_test_tabular_features.npy")
    # X_val = np.load(echonext_path / "EchoNext_val_tabular_features.npy")
    X_train = np.load(echonext_path / "EchoNext_train_tabular_features.npy")

    os.makedirs(output_path, exist_ok=False)

    if target_type == "composite":
        short_names = COMPOSITE_TARGET_NAMES
    elif target_type == "multitask":
        short_names = MULTITASK_TARGET_NAMES
    else:
        raise ValueError(f"Unknown target_type = {target_type}")
    target_cols = list(short_names.values())

    df = df.rename(columns=short_names)
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    train_targets = df.loc[train_mask, target_cols]
    test_targets = df.loc[test_mask, target_cols]

    targets = dict()
    for target_col in tqdm(target_cols):
        y_train = train_targets[target_col].to_numpy()
        y_test = test_targets[target_col].to_numpy()

        model = LogisticRegressionCV(
            Cs=10,
            l1_ratios=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
            penalty="elasticnet",
            cv=5,
            solver="saga",
            class_weight="balanced" if balance_class_weight else None,
            random_state=42,
            max_iter=100,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]
        targets[target_col] = y_pred
        targets[target_col + " (TRUE)"] = y_test

    multilabel_true = []
    multilabel_pred = []
    metrics = defaultdict(dict)
    for target_col in target_cols:
        y_test = targets[target_col + " (TRUE)"]
        y_pred = targets[target_col]

        metrics[target_col]["AUROC"] = roc_auc_score(y_test, y_pred)
        metrics[target_col]["AUPRC"] = average_precision_score(y_test, y_pred)

        multilabel_true.append(y_test)
        multilabel_pred.append(y_pred)

    multilabel_true = np.asarray(multilabel_true).T
    multilabel_pred = np.asarray(multilabel_pred).T
    auroc = roc_auc_score(multilabel_true, multilabel_pred, average="macro")
    auprc = average_precision_score(multilabel_true, multilabel_pred, average="macro")
    metrics["Multilabel Averaged"]["AUROC"] = auroc
    metrics["Multilabel Averaged"]["AUPRC"] = auprc
    metrics = pd.DataFrame.from_dict(metrics, orient="index")
    metrics.index.name = "Label"

    metrics.to_csv(os.path.join(output_path, "test-metrics.csv"))
    np.savez(os.path.join(output_path, "test-targets.npz"), **targets)


if __name__ == "__main__":
    args = parse_args()
    main(
        echonext_data=args.echonext_data,
        target_type=args.target_type,
        balance_class_weight=args.balance_class_weight,
        output_path=args.output_path,
    )
