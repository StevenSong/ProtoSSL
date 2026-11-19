"""Apply a finished PILA experiment's linear probe to another dataset's embeddings.

`compute-embeddings` run with the source experiment's `proj.ckpt` gives the target
dataset's similarities to the source's prototypes. This script standardizes them the
way `_linear_probe.py` did for the source (a `StandardScaler` fit on the source's train
embeddings, which `model.joblib` doesn't store) and predicts the source's labels, i.e.
imputes e.g. PTB-XL's statements on EchoNext.

Before applying anything to the target, the rebuilt scaler and the probe must
reproduce the source's own test predictions (`probs.npy`).

Writes `<split>_probs.npy` (N, n_source_labels) for the train and test splits, and
`label_names.json` with the source labels in column order.
"""

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from protossl.datasets import infer_dataset_class_from_path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--source-dataset-path", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def predict(model, X: np.ndarray) -> np.ndarray:
    """Positive-class probabilities, (N, n_labels), as `_linear_probe.py` saves them."""
    return np.asarray([p[:, 1] for p in model.predict_proba(X)]).T


def main(
    *,  # enforce kwargs
    source_experiment: str,
    source_dataset_path: str,
    embeddings: str,
    output_path: str,
):
    source = Path(source_experiment)
    source_embeds = source / "compute-embeddings/latest"
    model = joblib.load(source / "model.joblib")
    _, label_names, _ = infer_dataset_class_from_path(source_dataset_path)
    assert label_names is not None
    assert len(model.estimators_) == len(label_names)

    scaler = StandardScaler().fit(np.load(source_embeds / "train_embeds.npy"))
    n_features = scaler.n_features_in_
    assert all(
        e.n_features_in_ == n_features for e in model.estimators_
    ), "probe was not fit on the raw embeddings (PCA?), can't transfer it"
    expected = np.load(source / "probs.npy")
    actual = predict(
        model, scaler.transform(np.load(source_embeds / "test_embeds.npy"))
    )
    max_diff = np.abs(actual - expected).max()
    assert (
        max_diff < 1e-6
    ), f"rebuilt probe doesn't reproduce source probs.npy: {max_diff}"
    print(f"rebuilt probe reproduces source test probs (max abs diff {max_diff:.2e})")

    out = Path(output_path)
    for split in ["train", "test"]:
        X = np.load(Path(embeddings) / f"{split}_embeds.npy")
        assert X.shape[1] == n_features, (split, X.shape, n_features)
        probs = predict(model, scaler.transform(X))
        np.save(out / f"{split}_probs.npy", probs)
        print(f"{split}: {probs.shape} probabilities")
    (out / "label_names.json").write_text(json.dumps(list(label_names)) + "\n")


if __name__ == "__main__":
    # joblib has some instability with /dev/shm so use a tmp folder
    os.environ.setdefault(
        "JOBLIB_TEMP_FOLDER", os.path.join(os.path.expanduser("~"), ".tmp")
    )
    args = parse_args()
    main(
        source_experiment=args.source_experiment,
        source_dataset_path=args.source_dataset_path,
        embeddings=args.embeddings,
        output_path=args.output_path,
    )
