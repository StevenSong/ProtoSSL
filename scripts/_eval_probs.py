import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TARGET_NAMES = {
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
    "shd_moderate_or_greater_flag": "SHD",
}
COMPOSITE_TARGET = "SHD"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--echonext-data", required=True)
    parser.add_argument("--probs-npy", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    echonext_data: str,
    probs_npy: str,
    output_path: str,
):
    echonext_path = Path(echonext_data)

    df = pd.read_csv(echonext_path / "EchoNext_metadata_100k.csv")
    df = df.rename(columns=TARGET_NAMES)

    target_probs = np.load(probs_npy, allow_pickle=True)

    os.makedirs(output_path, exist_ok=True)
    assert not os.path.exists(os.path.join(output_path, "metrics.csv"))

    target_cols = list(TARGET_NAMES.values())

    test_mask = df["split"] == "test"

    test_targets = df.loc[test_mask, target_cols]

    multilabel_true = []
    multilabel_prob = []
    composite_true = None
    composite_prob = None
    metrics = defaultdict(dict)
    for i, target_col in enumerate(target_cols):
        y_test = test_targets[target_col].to_numpy()
        y_prob = target_probs[:, i]

        if target_col != COMPOSITE_TARGET:
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

    assert composite_true is not None and composite_prob is not None
    metrics[COMPOSITE_TARGET]["AUROC"] = roc_auc_score(composite_true, composite_prob)
    metrics[COMPOSITE_TARGET]["AUPRC"] = average_precision_score(
        composite_true, composite_prob
    )

    metrics = pd.DataFrame.from_dict(metrics, orient="index")
    metrics.index.name = "Label"

    metrics.to_csv(os.path.join(output_path, "metrics.csv"))


if __name__ == "__main__":
    args = parse_args()
    main(
        echonext_data=args.echonext_data,
        probs_npy=args.probs_npy,
        output_path=args.output_path,
    )
