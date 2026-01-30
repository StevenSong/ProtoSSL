import argparse
import os

import joblib
import numpy as np
import pandas as pd
from optuna import Study

from pass_pclr.defines import PTBXL_CAT1_TARGETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--study-pkl", required=True)
    parser.add_argument("--trial-predictions")
    parser.add_argument("--trial-checkpoints")
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    output_path: str,
    study_pkl: str,
    trial_predictions: str | None = None,
    trial_checkpoints: str | None = None,
):
    assert (
        trial_predictions is not None or trial_checkpoints is not None
    ), "Must provide at least one of trial_predictions or trial_checkpoints to postprocess"

    mapping = {k: f"Prob_{k}" for k in PTBXL_CAT1_TARGETS}
    cols = list(mapping.values())
    optuna_study: Study = joblib.load(study_pkl)
    best_trial = optuna_study.best_trial.number

    if trial_predictions is not None:
        best_csv = os.path.join(
            trial_predictions,
            f"trial_{best_trial}",
            f"trial_{best_trial}_test_results_v0.csv",
        )
        results = pd.read_csv(best_csv, index_col=0)[cols].rename(columns=mapping)
        probs_path = os.path.join(output_path, "probs.npy")
        np.save(probs_path, results.to_numpy())
        print(f"Saved predictions to {probs_path}")

    if trial_checkpoints is not None:
        checkpoint_dir = os.path.join(trial_checkpoints, f"trial_{best_trial}")
        checkpoints = os.listdir(checkpoint_dir)
        n_ckpts = len(checkpoints)
        assert (
            n_ckpts == 1
        ), f"Unexpected number of files ({n_ckpts}) in {checkpoint_dir}"
        ckpt_link = os.path.join(output_path, "best.ckpt")
        os.symlink(
            src=os.path.join(f"trial_{best_trial}", checkpoints[0]),
            dst=ckpt_link,
        )
        print(f"Linked checkpoint to {ckpt_link}")


if __name__ == "__main__":
    args = parse_args()
    main(
        output_path=args.output_path,
        study_pkl=args.study_pkl,
        trial_predictions=args.trial_predictions,
        trial_checkpoints=args.trial_checkpoints,
    )
