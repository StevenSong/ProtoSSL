#!/usr/bin/env python3

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pass_pclr.datasets import infer_dataset_class_from_path
from pass_pclr.models._prototype_assigner import PrototypeAssigner
from pass_pclr.models._prototype_ilp_assigner import (
    build_association_matrix,
    solve_assignment_ilp,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"[OK] {msg}")


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def tensor_stats(x: torch.Tensor) -> str:
    x = x.detach().float()
    return (
        f"shape={tuple(x.shape)} "
        f"dtype={x.dtype} "
        f"min={x.min().item():.6f} "
        f"max={x.max().item():.6f} "
        f"mean={x.mean().item():.6f} "
        f"std={x.std().item():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity check ILP/effect-size prototype assignment."
    )

    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--sampling-rate", type=int, default=32000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/gpfs/data/bbj-lab/users/sethis/ecg-prototype-fm/results_audioset_proto_contrastive/pass-audioset-ssl-assign-sanity/learn-prototypes/v2-7ca4qlcl/best.ckpt",
    )

    # Must match the SSL checkpoint architecture
    parser.add_argument("--resnet-type", type=str, default="resnet18")
    parser.add_argument("--conv-type", type=str, default="PANNS")
    parser.add_argument("--audio-backbone-name", type=str, default="Cnn14")
    parser.add_argument("--prototype-type", type=str, default="partial")
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--partial-len", type=int, default=3200)
    parser.add_argument("--partial-overlap", type=float, default=0.5)
    parser.add_argument("--prototype-h", type=int, default=1)
    parser.add_argument("--prototype-w", type=int, default=1)

    parser.add_argument("--n-prototypes", type=int, required=True)
    parser.add_argument("--n-prototypes-per-label", type=int, default=5)

    # ILP / effect-size options
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--trim", type=float, default=0.10)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--n-neg-repeats", type=int, default=1)
    parser.add_argument("--balanced-negative-sampling", action="store_true")
    parser.add_argument("--time-limit-s", type=float, default=30.0)

    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)

    print_header("CONFIG")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"resolved_device: {device}")

    print_header("1) RESOLVE DATASET")
    DatasetCls, label_names = infer_dataset_class_from_path(args.dataset_path)
    expect(label_names is not None, "dataset exposes label names")
    n_labels = len(label_names)
    print(f"DatasetCls: {DatasetCls}")
    print(f"n_labels: {n_labels}")

    expected_min_prototypes = n_labels * args.n_prototypes_per_label
    print(f"expected_min_prototypes={expected_min_prototypes}")
    print(f"provided_n_prototypes={args.n_prototypes}")

    expect(
        args.n_prototypes >= expected_min_prototypes,
        "n_prototypes is at least n_labels * n_prototypes_per_label",
    )

    ds = DatasetCls(
        dataset_path=args.dataset_path,
        split=args.split,
        sampling_rate=args.sampling_rate,
    )
    expect(len(ds) > 0, "dataset split is non-empty")

    loader = DataLoader(
        ds,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print_header("2) BUILD PROTOTYPE ASSIGNER FROM SSL CHECKPOINT")
    model = PrototypeAssigner(
        resnet_type=args.resnet_type,
        conv_type=args.conv_type,
        prototype_type=args.prototype_type,
        assignment_strategy="ilp_effect_size",
        n_prototypes=args.n_prototypes,
        n_prototypes_per_label=args.n_prototypes_per_label,
        n_binary_labels=n_labels,
        pretrained_weights=args.checkpoint_path,
        input_channels=args.input_channels,
        partial_len=args.partial_len,
        partial_overlap=args.partial_overlap,
        prototype_h=args.prototype_h,
        prototype_w=args.prototype_w,
        audio_backbone_name=args.audio_backbone_name,
    ).to(device)

    model.eval()
    print(model.__class__.__name__)
    print(f"encoder prototype table shape: {tuple(model.encoder.prototypes.shape)}")

    expect(
        model.encoder.prototypes.shape[0] == args.n_prototypes,
        "loaded prototype table has expected number of prototypes",
    )

    print_header("3) COLLECT RAW PROTOTYPE ACTIVATIONS")
    all_acts = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            waveform = batch["waveform"].to(device, non_blocking=True)
            labels = batch["label"].detach().cpu()

            acts = model.encoder.forward_raw_prototypes(waveform).detach().cpu()

            if batch_idx == 0:
                print("first batch raw activations:", tensor_stats(acts))
                print("first batch labels:", tensor_stats(labels.float()))

            all_acts.append(acts)
            all_labels.append(labels)

    A = torch.cat(all_acts, dim=0).numpy()
    Y = torch.cat(all_labels, dim=0).numpy()

    print(f"A.shape={A.shape}")
    print(f"Y.shape={Y.shape}")

    expect(A.ndim == 2, "A is 2D")
    expect(Y.ndim == 2, "Y is 2D")
    expect(A.shape[0] == Y.shape[0], "A and Y have same number of samples")
    expect(A.shape[1] == args.n_prototypes, "A has expected number of prototype columns")
    expect(Y.shape[1] == n_labels, "Y has expected number of label columns")

    print_header("4) BUILD ASSOCIATION MATRIX")
    association_matrix, valid_class_mask = build_association_matrix(
        A,
        Y,
        n_min=args.n_min,
        trim=args.trim,
        eps=args.eps,
        n_neg_repeats=args.n_neg_repeats,
        balanced_negative_sampling=args.balanced_negative_sampling,
        random_seed=args.seed,
    )

    print(f"association_matrix.shape={association_matrix.shape}")
    print(f"valid_class_mask.shape={valid_class_mask.shape}")
    print(f"num_valid_classes={int(valid_class_mask.sum())}/{len(valid_class_mask)}")

    expect(
        association_matrix.shape == (args.n_prototypes, n_labels),
        "association matrix has shape (P, C)",
    )
    expect(
        valid_class_mask.shape == (n_labels,),
        "valid class mask has shape (C,)",
    )
    expect(
        valid_class_mask.all(),
        "all classes are valid for this sanity run",
    )

    print_header("5) SOLVE ILP")
    result = solve_assignment_ilp(
        association_matrix,
        n_prototypes_per_label=args.n_prototypes_per_label,
        valid_class_mask=valid_class_mask,
        time_limit_s=args.time_limit_s,
    )

    selected = result.selected_indices_by_class
    assignment_matrix = result.assignment_matrix

    print(f"selected_indices_by_class.shape={selected.shape}")
    print(f"assignment_matrix.shape={assignment_matrix.shape}")
    print(f"objective_value={result.objective_value:.6f}")

    expect(
        selected.shape == (n_labels, args.n_prototypes_per_label),
        "selected indices have shape (C, K)",
    )
    expect(
        assignment_matrix.shape == (args.n_prototypes, n_labels),
        "assignment matrix has shape (P, C)",
    )

    col_sums = assignment_matrix.sum(axis=0)
    row_sums = assignment_matrix.sum(axis=1)

    print(f"column sums (first 20): {col_sums[:20]}")
    print(f"row sums stats: min={row_sums.min()} max={row_sums.max()} mean={row_sums.mean():.6f}")

    expect(
        np.all(col_sums == args.n_prototypes_per_label),
        "each class received exactly K prototypes",
    )
    expect(
        np.all(row_sums <= 1),
        "each prototype assigned to at most one class",
    )
    expect(
        assignment_matrix.sum() == n_labels * args.n_prototypes_per_label,
        "total number of assigned prototypes is exactly C*K",
    )

    used_prototypes = int((row_sums == 1).sum())
    unused_prototypes = int((row_sums == 0).sum())

    print(f"used_prototypes={used_prototypes}")
    print(f"unused_prototypes={unused_prototypes}")

    expect(
        used_prototypes == n_labels * args.n_prototypes_per_label,
        "number of used prototypes matches C*K",
    )

    print_header("6) CHECK ASSIGNED SCORES LOOK PLAUSIBLE")
    assigned_scores = []
    for c in range(n_labels):
        for k in selected[c]:
            assigned_scores.append(float(association_matrix[int(k), c]))
    assigned_scores = np.asarray(assigned_scores)

    print(
        f"assigned score stats: "
        f"min={assigned_scores.min():.6f} "
        f"max={assigned_scores.max():.6f} "
        f"mean={assigned_scores.mean():.6f} "
        f"std={assigned_scores.std():.6f}"
    )

    print_header("7) CONVERT TO PROTOTYPECLASSIFIER")
    indices_t = torch.as_tensor(selected, dtype=torch.long, device=model.encoder.prototypes.device)
    with torch.no_grad():
        assigned_model = model.convert_to_proto_classifier_from_indices(indices_t)

    print(assigned_model.__class__.__name__)
    expect(
        assigned_model.encoder.prototypes.shape[0] == expected_min_prototypes,
        "assigned classifier has exactly C*K prototypes",
    )

    print_header("8) SAVE OUTPUTS")
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for c in range(n_labels):
        for slot_idx in range(args.n_prototypes_per_label):
            prot_idx = int(selected[c, slot_idx])
            rows.append(
                {
                    "label_idx": c,
                    "label_name": label_names[c],
                    "slot_idx": slot_idx,
                    "prototype_idx": prot_idx,
                    "association_score": float(association_matrix[prot_idx, c]),
                }
            )

    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "assignment_metadata.csv"),
        index=False,
    )
    np.save(os.path.join(args.output_dir, "association_matrix.npy"), association_matrix)
    np.save(os.path.join(args.output_dir, "assignment_matrix.npy"), assignment_matrix)
    np.save(os.path.join(args.output_dir, "selected_indices_by_class.npy"), selected)
    np.save(os.path.join(args.output_dir, "valid_class_mask.npy"), valid_class_mask)

    ckpt_out = os.path.join(args.output_dir, "assigned_sanity.ckpt")
    torch.save({"state_dict": assigned_model.state_dict()}, ckpt_out)

    print(f"saved assignment_metadata.csv to {args.output_dir}")
    print(f"saved assigned_sanity.ckpt to {ckpt_out}")

    print_header("DONE")
    print("ILP assignment sanity checks completed.")


if __name__ == "__main__":
    main()