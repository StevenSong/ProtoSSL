import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 24))
torch.set_num_threads(n_threads)
torch.set_num_interop_threads(n_threads)

from protossl.datasets import infer_dataset_class_from_path
from protossl.models.helpers import build_association_matrix, solve_assignment_ilp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--patch-embeddings", required=True)
    parser.add_argument("--prototypes-per-label", type=int, required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main(
    *,  # enforce kwargs
    dataset_path: str,
    patch_embeddings: str,
    prototypes_per_label: int,
    output_path: str,
    random_seed: int = 42,
):
    ds_cls, _, is_audio = infer_dataset_class_from_path(dataset_path)
    assert not is_audio, "ECG only script"

    K = prototypes_per_label

    # load patch level embeddings
    X_train = np.load(os.path.join(patch_embeddings, "train_embeds.npy"))
    X_test = np.load(os.path.join(patch_embeddings, "test_embeds.npy"))

    # (N = n_samples, L = n_patches, H = emb_dim)
    X_train = torch.as_tensor(X_train.mT).contiguous()
    X_test = torch.as_tensor(X_test.mT).contiguous()

    # get corresponding labels
    ds = ds_cls(
        dataset_path=dataset_path,
        split="train",
        sampling_rate=100,  # doesn't matter, we're not using it
    )
    assert ds.labels is not None
    y_train = ds.labels  # (N, C = n_classes)

    N, L, H = X_train.shape
    _N, C = y_train.shape
    assert N == _N

    rng = torch.manual_seed(random_seed)
    protos = nn.init.trunc_normal_(  # (P, H)
        torch.empty(C * K, H), std=0.02, generator=rng
    )

    protos_norm = F.normalize(protos, dim=-1)
    X_norm = F.normalize(X_train, dim=-1)

    sims = torch.einsum("nlh,ph->nlp", X_norm, protos_norm)  # (N, L, P)
    sims = sims.max(dim=1).values  # (N, P)

    Q, valid_class_mask = build_association_matrix(
        sims.numpy(),
        y_train.numpy(),
        n_min=1,  # min positive samples
        trim=0.10,
        eps=1e-6,
        n_neg_repeats=10,  # number of resample repeats
        balanced_negative_sampling=True,
        weight_effects_using_lr=None,
    )
    result = solve_assignment_ilp(
        Q,
        n_prototypes_per_label=K,
        max_classes_per_prototype=1,
        valid_class_mask=valid_class_mask,
    )
    idxs = result.selected_indices_by_class  # (C, K)
    assert len(set(idxs.flatten())) == C * K
    protos = protos[idxs.flatten()].view(C, K, H)  # (C, K, H)

    # project prototypes to be real samples
    protos_norm = F.normalize(protos, dim=-1)  # (C, K, H)
    X_norm = F.normalize(X_train, dim=-1)  # (N, L, H)
    sims = torch.einsum("nlh,ckh->nlck", X_norm, protos_norm)  # (N, L, C, K)

    # only consider samples that have positive label
    mask = y_train[:, None, :, None].bool()  # (N, 1, C, 1)
    sims = sims.masked_fill(~mask, -torch.inf)
    assert (
        (~torch.isinf(sims))  # (N, L, C, K) - True where valid
        .any(dim=0)  # (L, C, K)
        .any(dim=0)  # (C, K)
        .all()  # ensure some valid entry to all C,K
    )

    sims_flat = sims.view(N * L, C, K)  # (NL, C, K)
    best_nl = sims_flat.argmax(dim=0)  # (C, K)
    X_flat = X_train.view(N * L, H)  # (NL, H)
    protos = X_flat[best_nl]  # (C, K, H)

    # compute prototype activations relative to patches
    protos = protos.view(-1, H)  # (CK, H)
    protos_norm = F.normalize(protos, dim=-1)
    X_train = F.normalize(X_train, dim=-1)
    X_test = F.normalize(X_test, dim=-1)
    X_train = torch.einsum("nlh,ph->nlp", X_train, protos_norm)  # (N, L, CK)
    X_test = torch.einsum("nlh,ph->nlp", X_test, protos_norm)
    X_train = X_train.max(dim=1).values.numpy()  # (N, CK)
    X_test = X_test.max(dim=1).values.numpy()

    # save prototype embeddings and metadata
    protos = protos.numpy()
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "train_embeds.npy"), X_train)
    np.save(os.path.join(output_path, "test_embeds.npy"), X_test)
    np.save(os.path.join(output_path, "protos.npy"), protos)


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset_path,
        patch_embeddings=args.patch_embeddings,
        prototypes_per_label=args.prototypes_per_label,
        output_path=args.output_path,
        random_seed=args.random_seed,
    )
