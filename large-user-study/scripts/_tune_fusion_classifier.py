"""
Tune the ProtoECGNetFusion classification head (stage 3) over precomputed
prototype similarities from `--pipeline-stage compute-fusion-embeddings`.

The encoders are frozen in train-fusion-classifier, so the head only ever sees
the prototype similarities. Training directly on the cached similarities is
numerically the same objective as ProtoECGNetFusion.forward but skips the
dataloaders and encoder forward passes, so an Optuna study is cheap to run.

Tuned: lam_l1 (masked L1 on off-class weights), lr, weight_decay, batch_size.
Fixed (matching configs/fusion.yaml): AdamW, ReduceLROnPlateau(val_loss,
factor=0.1, patience=3), bias-free linear head with masked init (1 / -0.5).

Model selection within a trial and across trials uses val macro AP. val_loss
includes the lam_l1-scaled penalty, so it is not comparable across lam_l1 values
(it is still used for the LR schedule within a trial, as in the Lightning config).
"""

import json
import os
from argparse import ArgumentParser

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchmetrics.functional.classification import (
    multilabel_auroc,
    multilabel_average_precision,
)

from protossl.datasets import (
    HeedbECGDataset,
    get_heedb_labels,
    get_heedb_metadata,
    infer_dataset_class_from_path,
)
from protossl.models._protoecgnet import BranchCfg, fusion_prototype_assignment

torch.set_float32_matmul_precision("medium")

# baseline from configs/fusion.yaml + ProtoECGNetFusion defaults, enqueued as trial 0
BASELINE_PARAMS = {"lam_l1": 1e-4, "lr": 1e-3, "weight_decay": 0.01, "batch_size": 512}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--embeds-dir",
        required=True,
        help="compute-fusion-embeddings log dir (has config.yaml and {split}_embeds.npy)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ap-tolerance",
        type=float,
        default=0.0,
        help=(
            "select the sparsest head (lowest off-class weight fraction) among "
            "trials within this val AP of the best trial; 0 = pure best val AP"
        ),
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="skip tuning, (re)select from existing study and export probs",
    )
    return parser.parse_args()


def load_run_config(embeds_dir: str) -> dict:
    c = OmegaConf.load(os.path.join(embeds_dir, "config.yaml"))
    assert c.pipeline_stage == "compute-fusion-embeddings", c.pipeline_stage
    data_args = c.data.init_args
    dataset_path: str = data_args.dataset_path
    ds_cls, label_names, _ = infer_dataset_class_from_path(dataset_path)
    if ds_cls != HeedbECGDataset:
        raise ValueError(f"Only HEEDB is supported, got: {ds_cls}")
    label_subset = data_args.label_subset
    if label_subset is not None:
        label_subset = list(label_subset)
        label_names = label_subset
    branches = [
        BranchCfg(**b)  # type: ignore
        for b in OmegaConf.to_container(c.model.init_args.branches)  # type: ignore
    ]
    return {
        "dataset_path": dataset_path,
        "data_kwargs": OmegaConf.to_container(data_args.data_kwargs),
        "label_subset": label_subset,
        "label_names": label_names,
        "assign": fusion_prototype_assignment(label_names, branches),  # (L, R)
    }


def load_labels(run_cfg: dict, split: str) -> torch.Tensor:
    # same labels as HeedbECGDataset but without instantiating waveforms
    data_kwargs = dict(run_cfg["data_kwargs"])
    split_kwargs = {}
    if "heedb_split_type" in data_kwargs:
        split_kwargs["heedb_split_type"] = data_kwargs.pop("heedb_split_type")
    unknown = set(data_kwargs) - {"label_src"}
    if len(unknown) > 0:
        raise ValueError(f"Unhandled data_kwargs: {unknown}")
    df = get_heedb_metadata(run_cfg["dataset_path"], **split_kwargs)
    df = df[df["split"] == split].reset_index(drop=True)
    labels = get_heedb_labels(
        run_cfg["dataset_path"], df, run_cfg["label_subset"], **data_kwargs
    )
    return torch.as_tensor(labels)


def load_embeds(
    embeds_dir: str,
    split: str,
    device: str,
    chunk_rows: int = 500_000,
) -> torch.Tensor:
    arr = np.load(os.path.join(embeds_dir, f"{split}_embeds.npy"), mmap_mode="r")
    print(f"loading {split} embeds {arr.shape} to {device}")
    out = torch.empty(arr.shape, dtype=torch.float32, device=device)
    for i in range(0, arr.shape[0], chunk_rows):
        out[i : i + chunk_rows].copy_(
            torch.from_numpy(np.array(arr[i : i + chunk_rows]))
        )
    return out


@torch.no_grad()
def compute_logits(W: torch.Tensor, X: torch.Tensor, chunk_rows: int = 65536):
    return torch.cat(
        [X[i : i + chunk_rows] @ W.T for i in range(0, len(X), chunk_rows)]
    )


@torch.no_grad()
def sparsity_stats(W: torch.Tensor, off_class_mask: torch.Tensor) -> dict:
    abs_W = W.abs()
    off_l1 = (abs_W * off_class_mask).sum()
    off_abs = abs_W[off_class_mask.bool()]
    return {
        "off_class_l1": off_l1.item(),
        "on_class_l1": (abs_W * (1 - off_class_mask)).sum().item(),
        # share of total weight mass on off-class connections (lower = more interpretable)
        "off_class_frac": (off_l1 / abs_W.sum()).item(),
        # Adam + L1 subgradient does not produce exact zeros, so report near-zero
        "off_class_near_zero_frac": (off_abs < 1e-3).float().mean().item(),
    }


def train_head(
    *,  # enforce kwargs
    params: dict,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    assign: torch.Tensor,
    pos_weight: torch.Tensor,
    max_epochs: int,
    early_stopping_patience: int,
    lr_patience: int,
    seed: int,
    trial: optuna.Trial | None = None,
) -> tuple[torch.Tensor, dict]:
    device = X_train.device
    n_train, n_labels = Y_train.shape
    off_class_mask = 1.0 - assign
    lam_l1, batch_size = params["lam_l1"], params["batch_size"]

    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    # same masked init as ProtoECGNetFusion.cls
    W = (assign - 0.5 * off_class_mask).clone().requires_grad_(True)  # (L, R)
    opt = torch.optim.AdamW([W], lr=params["lr"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.1, patience=lr_patience
    )

    best_ap, best_epoch, best_W, best_metrics = -1.0, -1, None, {}
    for epoch in range(max_epochs):
        perm = torch.randperm(n_train, device=device, generator=gen)
        train_loss_sum = torch.zeros((), device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            logits = X_train[idx] @ W.T  # (B, L)
            cls_loss = F.binary_cross_entropy_with_logits(
                logits, Y_train[idx], pos_weight=pos_weight, reduction="mean"
            )
            penalty = (W.abs() * off_class_mask).sum()
            loss = cls_loss + lam_l1 * penalty
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss_sum += loss.detach() * len(idx)

        with torch.no_grad():
            val_logits = compute_logits(W, X_val)
            val_cls_loss = F.binary_cross_entropy_with_logits(
                val_logits, Y_val, pos_weight=pos_weight, reduction="mean"
            ).item()
            val_penalty = (lam_l1 * (W.abs() * off_class_mask).sum()).item()
            val_probs = val_logits.sigmoid()
            val_ap = multilabel_average_precision(
                val_probs, Y_val.int(), num_labels=n_labels, average="macro"
            ).item()
            val_auroc = multilabel_auroc(
                val_probs, Y_val.int(), num_labels=n_labels, average="macro"
            ).item()
        val_loss = val_cls_loss + val_penalty
        sched.step(val_loss)
        print(
            f"  epoch {epoch:3d} | train_loss {(train_loss_sum / n_train).item():.4f} "
            f"| val_loss {val_loss:.4f} (cls {val_cls_loss:.4f}) "
            f"| val_ap {val_ap:.4f} | val_auroc {val_auroc:.4f} "
            f"| lr {opt.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_W = W.detach().clone()
            best_metrics = {
                "val_ap": val_ap,
                "val_auroc": val_auroc,
                "val_classification_loss": val_cls_loss,
                "val_loss": val_loss,
                "best_epoch": epoch,
            }
        elif epoch - best_epoch >= early_stopping_patience:
            break

        if trial is not None:
            trial.report(val_ap, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    assert best_W is not None
    best_metrics["n_epochs"] = epoch + 1
    best_metrics.update(sparsity_stats(best_W, off_class_mask))
    return best_W, best_metrics


def tune(args, run_cfg: dict, study: optuna.Study, trials_dir: str):
    device = args.device
    assign = run_cfg["assign"].to(device)

    Y_train = load_labels(run_cfg, "train")
    Y_val = load_labels(run_cfg, "val")
    # same as BaseTSDataset.get_label_weights over the train split
    per_label_count = Y_train.sum(dim=0)
    pos_weight = ((Y_train.shape[0] - per_label_count) / per_label_count).float()

    X_train = load_embeds(args.embeds_dir, "train", device)
    X_val = load_embeds(args.embeds_dir, "val", device)
    for X, Y in [(X_train, Y_train), (X_val, Y_val)]:
        assert X.shape[0] == Y.shape[0], f"{X.shape} vs {Y.shape}"
        assert X.shape[1] == assign.shape[1], f"{X.shape} vs {assign.shape}"
    Y_train, Y_val = Y_train.float().to(device), Y_val.float().to(device)
    pos_weight = pos_weight.to(device)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lam_l1": trial.suggest_float("lam_l1", 1e-8, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 3e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [512, 2048, 8192]),
        }
        print(f"trial {trial.number}: {params}", flush=True)
        W, metrics = train_head(
            params=params,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            assign=assign,
            pos_weight=pos_weight,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.early_stopping_patience,
            lr_patience=args.lr_patience,
            seed=args.seed + trial.number,
            trial=trial,
        )
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        torch.save(
            {"cls.weight": W.cpu()}, os.path.join(trials_dir, f"{trial.number}.pt")
        )
        print(f"trial {trial.number} done: {metrics}", flush=True)
        return metrics["val_ap"]

    if len(study.trials) == 0:
        study.enqueue_trial(BASELINE_PARAMS)
    n_finished = len(
        [t for t in study.trials if t.state.is_finished()]
    )  # resume toward n_trials total
    study.optimize(objective, n_trials=max(args.n_trials - n_finished, 0))


def export(args, run_cfg: dict, study: optuna.Study, trials_dir: str):
    trials = study.trials_dataframe(
        attrs=("number", "state", "value", "params", "user_attrs")
    )
    trials = trials.rename(
        columns=lambda c: c.removeprefix("params_").removeprefix("user_attrs_")
    )
    trials = trials.sort_values("value", ascending=False)
    trials.to_csv(os.path.join(args.output_dir, "trials.csv"), index=False)

    done = trials[trials["state"] == "COMPLETE"]
    if len(done) == 0:
        raise ValueError("No completed trials to select from")
    best_ap = done["value"].max()
    candidates = done[done["value"] >= best_ap - args.ap_tolerance]
    selected = candidates.sort_values("off_class_frac").iloc[0]
    summary_cols = [
        "number", "value", "val_auroc", "lam_l1", "lr", "weight_decay",
        "batch_size", "off_class_frac", "off_class_near_zero_frac", "best_epoch",
    ]  # fmt: skip
    print(done[summary_cols].head(20).to_string(index=False))
    print(
        f"selected trial {int(selected['number'])} (ap tolerance {args.ap_tolerance})"
    )

    W = torch.load(os.path.join(trials_dir, f"{int(selected['number'])}.pt"))[
        "cls.weight"
    ]
    W = W.to(args.device)
    torch.save({"cls.weight": W.cpu()}, os.path.join(args.output_dir, "cls.pt"))
    for split in ["val", "test"]:
        X = load_embeds(args.embeds_dir, split, args.device)
        assert X.shape[1] == W.shape[1], f"{X.shape} vs {W.shape}"
        # column order is label_names, same layout as PredictionWriter probs
        probs = compute_logits(W, X).sigmoid().cpu().numpy()
        np.save(os.path.join(args.output_dir, f"{split}_probs.npy"), probs)
        del X

    selection = {
        "trial": int(selected["number"]),
        "ap_tolerance": args.ap_tolerance,
        "best_val_ap": float(best_ap),
        "label_names": run_cfg["label_names"],
        **{
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in selected.items()
            if k not in {"number", "state"}
        },
    }
    with open(os.path.join(args.output_dir, "selected.json"), "w") as f:
        json.dump(selection, f, indent=2, default=str)
    print(f"exported selected head and probs to {args.output_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    trials_dir = os.path.join(args.output_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)

    run_cfg = load_run_config(args.embeds_dir)
    study = optuna.create_study(
        study_name="fusion-classifier",
        storage=f"sqlite:///{os.path.join(args.output_dir, 'study.db')}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    if not args.export_only:
        tune(args, run_cfg, study, trials_dir)
        torch.cuda.empty_cache()  # release train embeds before export
    export(args, run_cfg, study, trials_dir)


if __name__ == "__main__":
    main()
