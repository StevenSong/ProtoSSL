import os
from argparse import ArgumentParser
from pathlib import Path
from warnings import simplefilter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pass_pclr.datasets import infer_dataset_class_from_path

simplefilter("ignore", category=ConvergenceWarning)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--prototype-embeddings", required=True)
    parser.add_argument("--embedding-pca", type=int)
    parser.add_argument("--balance-class-weight", action="store_true")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    dataset_path: str,
    prototype_embeddings: str,
    embedding_pca: int | None = None,
    balance_class_weight: bool,
    output_path: str,
):
    ds_cls, label_names = infer_dataset_class_from_path(dataset_path)
    assert label_names is not None

    if "audioset" in dataset_path.lower():
        sampling_rate = 32000
        print("Using audio sampling rate")
    else: 
        sampling_rate = 100
        print("Using ECG sampling rate")
    train_ds = ds_cls(dataset_path=dataset_path, split="train", sampling_rate=sampling_rate)

    assert train_ds.labels is not None
    train_targets = train_ds.labels.numpy()

    prototype_path = Path(prototype_embeddings)
    X_train = np.load(prototype_path / "train_embeds.npy")
    X_test = np.load(prototype_path / "test_embeds.npy")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if embedding_pca is not None:
        pca = PCA(n_components=embedding_pca, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    target_probs = []
    models = dict()
    for i, target_col in enumerate(tqdm(label_names)):
        y_train = train_targets[:, i]

        # model = LogisticRegressionCV(
        #     Cs=10,
        #     l1_ratios=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
        #     penalty="elasticnet",
        #     cv=5,
        #     solver="saga",
        #     class_weight="balanced" if balance_class_weight else None,
        #     random_state=42,
        #     max_iter=100,
        #     n_jobs=-1,
        # )
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced" if balance_class_weight else None,
            random_state=42,
            max_iter=100,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        models[target_col] = model

        y_prob = model.predict_proba(X_test)[:, 1]
        target_probs.append(y_prob)
    target_probs = np.asarray(target_probs).T
    np.save(os.path.join(output_path, "probs.npy"), target_probs)
    np.savez(os.path.join(output_path, "models.npz"), **models)


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset_path,
        prototype_embeddings=args.prototype_embeddings,
        embedding_pca=args.embedding_pca,
        balance_class_weight=args.balance_class_weight,
        output_path=args.output_path,
    )
