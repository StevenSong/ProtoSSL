import os

# joblib has some instability with /dev/shm so use a tmp folder
os.environ["JOBLIB_TEMP_FOLDER"] = os.path.join(os.path.expanduser("~"), ".tmp")

from argparse import ArgumentParser
from pathlib import Path
from warnings import simplefilter

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from pass_pclr.datasets import infer_dataset_class_from_path

simplefilter("ignore", category=ConvergenceWarning)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--prototype-embeddings", required=True)
    parser.add_argument("--embedding-pca", type=int)
    parser.add_argument("--balance-class-weight", action="store_true")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--label-subset", nargs="+")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    dataset_path: str,
    prototype_embeddings: str,
    embedding_pca: int | None = None,
    balance_class_weight: bool,
    output_path: str,
    label_subset: list[str] | None = None,
    random_seed: int = 42,
):
    ds_cls, label_names = infer_dataset_class_from_path(dataset_path)
    assert label_names is not None

    train_ds = ds_cls(
        dataset_path=dataset_path,
        split="train",
        sampling_rate=100,
        label_subset=label_subset,
    )

    assert train_ds.labels is not None
    train_targets = train_ds.labels.numpy()

    prototype_path = Path(prototype_embeddings)
    X_train = np.load(prototype_path / "train_embeds.npy")
    X_test = np.load(prototype_path / "test_embeds.npy")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if embedding_pca is not None:
        pca = PCA(n_components=embedding_pca, random_state=random_seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    model = MultiOutputClassifier(
        estimator=LogisticRegression(
            C=5e-4,
            penalty="l2",
            solver="saga",
            class_weight="balanced" if balance_class_weight else None,
            random_state=random_seed,
            max_iter=100,
        ),
        n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", -1)),
    )
    model.fit(X_train, train_targets)
    target_probs = [y_probs[:, 1] for y_probs in model.predict_proba(X_test)]

    target_probs = np.asarray(target_probs).T
    np.save(os.path.join(output_path, "probs.npy"), target_probs)
    joblib.dump(model, os.path.join(output_path, "model.joblib"))


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset_path,
        prototype_embeddings=args.prototype_embeddings,
        embedding_pca=args.embedding_pca,
        balance_class_weight=args.balance_class_weight,
        output_path=args.output_path,
        label_subset=args.label_subset,
        random_seed=args.random_seed,
    )
