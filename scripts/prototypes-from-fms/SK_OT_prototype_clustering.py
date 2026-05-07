import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 24))
torch.set_num_threads(n_threads)
torch.set_num_interop_threads(n_threads)

from protossl.datasets import infer_dataset_class_from_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--patch-embeddings", required=True)
    parser.add_argument("--prototypes-per-label", type=int, required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--n-sk-iters", type=int, default=3)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def solve_assignment(S_c: torch.Tensor, n_iterations=3, epsilon=0.05):
    # S_c is the similarities of positives samples to prototypes for a given class
    B_c, L, K = S_c.shape
    S_c_flat = S_c.view(B_c * L, K)  # (B_c*L, K)
    A_c = torch.exp(S_c_flat / epsilon).t()  # (K, B_c*L)
    BcL = B_c * L

    # make the matrix sums to 1
    sum_A_c = torch.sum(A_c)
    A_c /= sum_A_c

    for _ in range(n_iterations):
        A_c /= torch.sum(A_c, dim=1, keepdim=True)
        A_c /= K

        A_c /= torch.sum(A_c, dim=0, keepdim=True)
        A_c /= BcL

    A_c *= BcL
    A_c = A_c.t()  # (B_c*L, K)

    indices = torch.argmax(A_c, dim=1)
    A_c = F.one_hot(indices, num_classes=K).to(dtype=torch.float32)

    return A_c.view(B_c, L, K)


def get_class_proto_update(
    *,  # enforce kwargs
    X_c: torch.Tensor,  # (B_c, L, H)
    protos_c: torch.Tensor,  # (K, H)
    beta: float,
    n_sk_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    K, _ = protos_c.shape
    new_protos_c = protos_c.clone()
    counter_c = torch.zeros(K)

    X_norm = F.normalize(X_c, dim=-1)  # (B_c, L, H)
    protos_norm = F.normalize(protos_c, dim=-1)  # (K, H)
    S_c = torch.einsum("blh,kh->blk", X_norm, protos_norm)  # (B_c, L, K)

    # do SK clustering to solve assignments
    A_c = solve_assignment(S_c, n_iterations=n_sk_iters)  # (B_c, L, K)

    for k in range(K):
        A_ck = A_c[:, :, k]  # (B_c, L)
        A_ck_mask = A_ck == 1
        if A_ck_mask.sum() == 0:
            # sometimes the SK doesn't converge perfectly and we get
            # prototype-slots which don't have any patches assigned, so
            # skip it but keep track of how many times each slot is updated
            continue
        counter_c[k] += 1
        cluster_mean = X_c[A_ck_mask].mean(dim=0)  # (H,)

        # momentum update
        new_protos_c[k] = beta * protos_c[k] + (1 - beta) * cluster_mean

    return new_protos_c, counter_c


def main(
    *,  # enforce kwargs
    dataset_path: str,
    patch_embeddings: str,
    prototypes_per_label: int,
    output_path: str,
    batch_size: int,
    n_epochs: int,
    n_sk_iters: int,
    momentum: float,
    random_seed: int = 42,
):
    ds_cls, _, is_audio = infer_dataset_class_from_path(dataset_path)
    assert not is_audio, "ECG only script"

    K = prototypes_per_label
    B = batch_size

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
    protos = nn.init.trunc_normal_(  # (C, K, H)
        torch.empty(C, K, H), std=0.02, generator=rng
    )

    # keep track of how many times we should and actually update prototypes
    ideal = torch.zeros(C, K)
    counter = torch.zeros(C, K)

    for _ in trange(n_epochs, desc="Epoch"):
        for i in trange(0, N, B, desc="Batch"):
            X_batch = X_train[i : i + B]
            y_batch = y_train[i : i + B]

            for c in range(C):
                mask = y_batch[:, c] == 1
                if mask.sum() > 0:
                    # ideally, should update all prototype-slots if we have instances in this batch
                    ideal[c] += 1
                    X_c = X_batch[mask]  # (B_c, L, H)
                    new_protos_c, counter_c = get_class_proto_update(
                        X_c=X_c,
                        protos_c=protos[c],
                        beta=momentum,
                        n_sk_iters=n_sk_iters,
                    )
                    protos[c] = new_protos_c
                    counter[c] += counter_c

    if (counter / ideal).min() == 0:
        print("=" * 40)
        print(f"!!!WARNING: {int((counter == 0).sum())} PROTOTYPES WERE NOT UPDATED!!!")
        print("=" * 40)

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
    assign_frec = (counter / ideal).numpy()
    protos = protos.numpy()
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "train_embeds.npy"), X_train)
    np.save(os.path.join(output_path, "test_embeds.npy"), X_test)
    np.save(os.path.join(output_path, "protos.npy"), protos)
    np.save(os.path.join(output_path, "assign_frec.npy"), assign_frec)


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.dataset_path,
        patch_embeddings=args.patch_embeddings,
        prototypes_per_label=args.prototypes_per_label,
        output_path=args.output_path,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_sk_iters=args.n_sk_iters,
        momentum=args.momentum,
        random_seed=args.random_seed,
    )
