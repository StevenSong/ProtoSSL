import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, get_args
from warnings import simplefilter

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

simplefilter("ignore", category=ConvergenceWarning)


EMB_T = Literal["sim1d", "sim2d_partial", "sim2d_global", "all"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--echonext-data", required=True)
    parser.add_argument("--prototype-embeddings", required=True)
    parser.add_argument("--embedding-type", required=True, choices=get_args(EMB_T))
    parser.add_argument("--embedding-pca", type=int)
    parser.add_argument("--balance-class-weight", action="store_true")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    target_config: str,
    echonext_data: str,
    prototype_embeddings: str,
    embedding_type: EMB_T,
    embedding_pca: int | None = None,
    balance_class_weight: bool,
    output_path: str,
):
    with open(target_config, "r") as f:
        config = yaml.safe_load(f)
        mapping = config["target_columns"]  # name --> col
        mapping = {v: k for k, v in mapping.items()}  # col --> name

    echonext_path = Path(echonext_data)
    df = pd.read_csv(echonext_path / "EchoNext_metadata_100k.csv")
    df = df.rename(columns=mapping)
    target_cols = list(mapping.values())
    train_mask = df["split"] == "train"
    train_targets = df.loc[train_mask, target_cols]

    prototype_path = Path(prototype_embeddings)
    if embedding_type == "sim1d":
        X_train = np.load(prototype_path / "train_sim1ds.npy")
        X_test = np.load(prototype_path / "test_sim1ds.npy")
    elif embedding_type == "sim2d_partial":
        X_train = np.load(prototype_path / "train_sim2d_partials.npy")
        X_test = np.load(prototype_path / "test_sim2d_partials.npy")
    elif embedding_type == "sim2d_global":
        X_train = np.load(prototype_path / "train_sim2d_globals.npy")
        X_test = np.load(prototype_path / "test_sim2d_globals.npy")
    elif embedding_type == "all":
        X_train_sim1d = np.load(prototype_path / "train_sim1ds.npy")
        X_test_sim1d = np.load(prototype_path / "test_sim1ds.npy")
        X_train_sim2d_partial = np.load(prototype_path / "train_sim2d_partials.npy")
        X_test_sim2d_partial = np.load(prototype_path / "test_sim2d_partials.npy")
        X_train_sim2d_global = np.load(prototype_path / "train_sim2d_globals.npy")
        X_test_sim2d_global = np.load(prototype_path / "test_sim2d_globals.npy")
        X_train = np.concat(
            [X_train_sim1d, X_train_sim2d_partial, X_train_sim2d_global], axis=1
        )
        X_test = np.concat(
            [X_test_sim1d, X_test_sim2d_partial, X_test_sim2d_global], axis=1
        )
    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if embedding_pca is not None:
        pca = PCA(n_components=embedding_pca, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    os.makedirs(output_path, exist_ok=False)

    target_probs = []
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

        y_prob = model.predict_proba(X_test)[:, 1]
        target_probs.append(y_prob)
    target_probs = np.asarray(target_probs).T
    np.save(os.path.join(output_path, "probs.npy"), target_probs)


if __name__ == "__main__":
    args = parse_args()
    main(
        target_config=args.target_config,
        echonext_data=args.echonext_data,
        prototype_embeddings=args.prototype_embeddings,
        embedding_type=args.embedding_type,
        embedding_pca=args.embedding_pca,
        balance_class_weight=args.balance_class_weight,
        output_path=args.output_path,
    )
