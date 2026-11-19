import os
from argparse import ArgumentParser
from pathlib import Path
from warnings import simplefilter

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from protossl.datasets import infer_dataset_class_from_path

simplefilter("ignore", category=ConvergenceWarning)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--prototype-embeddings", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    dataset_path: str,
    prototype_embeddings: str,
    output_path: str,
    random_seed: int = 42,
):
    ds_cls, label_names, is_audio = infer_dataset_class_from_path(dataset_path)
    if not is_audio:
        raise ValueError("This eval script is meant for audio related work")
    assert label_names is not None

    train_ds = ds_cls(
        dataset_path=dataset_path,
        split="train",
        sampling_rate=32000,
    )

    assert train_ds.labels is not None
    y_train = train_ds.labels.numpy()  # (N, n_labels)

    # convert one-hot labels to multiclass
    assert y_train.ndim == 2, "Expected 2D matrix (N, num_classes)"
    assert (y_train.sum(axis=-1) == 1).all(), "Rows must sum to 1"
    assert (y_train >= 0).all() and (y_train <= 1).all(), "Values must be 0 or 1"
    y_train = y_train.argmax(axis=-1)

    prototype_path = Path(prototype_embeddings)
    X_train = np.load(prototype_path / "train_embeds.npy")
    X_test = np.load(prototype_path / "test_embeds.npy")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # multinomial multiclass
    model = LogisticRegression(
        C=5e-4,
        penalty="l2",
        solver="saga",
        random_state=random_seed,
        max_iter=100,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)  # (N, n_labels)

    np.save(os.path.join(output_path, "probs.npy"), y_prob)
    joblib.dump(model, os.path.join(output_path, "model.joblib"))


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset_path,
        prototype_embeddings=args.prototype_embeddings,
        output_path=args.output_path,
        random_seed=args.random_seed,
    )
