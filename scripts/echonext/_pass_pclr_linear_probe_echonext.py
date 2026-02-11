import os
from argparse import ArgumentParser
from pathlib import Path
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pass_pclr.defines import ECHONEXT_TARGETS

simplefilter("ignore", category=ConvergenceWarning)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--echonext-data", required=True)
    parser.add_argument("--prototype-embeddings", required=True)
    parser.add_argument("--embedding-pca", type=int)
    parser.add_argument("--balance-class-weight", action="store_true")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    echonext_data: str,
    prototype_embeddings: str,
    embedding_pca: int | None = None,
    balance_class_weight: bool,
    output_path: str,
):
    mapping = {v: k for k, v in ECHONEXT_TARGETS.items()}

    echonext_path = Path(echonext_data)
    df = pd.read_csv(echonext_path / "EchoNext_metadata_100k.csv")
    df = df.rename(columns=mapping)
    target_cols = list(mapping.values())
    train_mask = df["split"] == "train"
    train_targets = df.loc[train_mask, target_cols]

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
    for target_col in tqdm(target_cols):
        y_train = train_targets[target_col].to_numpy()

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
        models[target_col] = model

        y_prob = model.predict_proba(X_test)[:, 1]
        target_probs.append(y_prob)
    target_probs = np.asarray(target_probs).T
    np.save(os.path.join(output_path, "probs.npy"), target_probs)
    np.savez(os.path.join(output_path, "models.npz"), **models)


if __name__ == "__main__":
    args = parse_args()
    main(
        echonext_data=args.echonext_data,
        prototype_embeddings=args.prototype_embeddings,
        embedding_pca=args.embedding_pca,
        balance_class_weight=args.balance_class_weight,
        output_path=args.output_path,
    )
