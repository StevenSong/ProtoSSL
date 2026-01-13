import os
import re
from pathlib import Path
from typing import Literal, Type, get_args

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader
from wandb.util import generate_id

from .datasets import (
    BaseECGDataset,
    EchoNextECGDataset,
    PCLRWrapperDataset,
    PtbxlECGDataset,
)
from .defines import ECHONEXT_TARGETS, RESNET_T, STAGE_T
from .models import PrototypeClassifier, PrototypeContraster, ResNetClassifier

MODEL_T = Literal[
    "PrototypeContraster",
    "PrototypeClassifier",
    "ResNetClassifier",
]

torch.set_float32_matmul_precision("medium")


def infer_dataset_class_from_path(
    dataset_path: str,
) -> tuple[
    Type[BaseECGDataset],
    list[str] | None,  # label names
]:
    echonext_indicators = ["echonext", "echo-next", "echo_next"]
    ptbxl_indicators = ["ptbxl", "ptb-xl", "ptb_xl"]

    if any(x in dataset_path for x in echonext_indicators):
        return EchoNextECGDataset, list(ECHONEXT_TARGETS.keys())
    elif any(x in dataset_path for x in ptbxl_indicators):
        return PtbxlECGDataset, None
    raise ValueError(
        f"Could not infer BaseECGDataset subclass from dataset_path: {dataset_path}"
    )


class LitData(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        pipeline_stage: STAGE_T,
        batch_size: int,
        num_workers: int,
        sampling_rate: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        dataset_path: str = self.hparams.dataset_path  # type: ignore
        sampling_rate: int = self.hparams.sampling_rate  # type: ignore
        pipeline_stage: STAGE_T = self.hparams.pipeline_stage  # type: ignore
        wrap_pclr = pipeline_stage == "learn-prototypes"

        ECGDataset, self.label_names = infer_dataset_class_from_path(dataset_path)

        if stage == "fit":
            train_ds = ECGDataset(
                dataset_path=dataset_path,
                split="train",
                sampling_rate=sampling_rate,
            )
            if wrap_pclr:
                train_ds = PCLRWrapperDataset(train_ds)
            self.train_ds = train_ds
        if stage in ["fit", "validate"]:
            val_ds = ECGDataset(
                dataset_path=dataset_path,
                split="val",
                sampling_rate=sampling_rate,
            )
            if wrap_pclr:
                val_ds = PCLRWrapperDataset(val_ds)
            self.val_ds = val_ds
        if stage in ["test", "predict"]:
            test_ds = ECGDataset(
                dataset_path=dataset_path,
                split="test",
                sampling_rate=sampling_rate,
            )
            if wrap_pclr:
                raise ValueError("Should not use PCLR dataset with test/predict stage")
            self.test_ds = test_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.hparams.num_workers,  # type: ignore
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.hparams.num_workers,  # type: ignore
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.hparams.num_workers,  # type: ignore
        )

    def predict_dataloader(self):
        return self.test_dataloader()


class LitModel(LightningModule):
    def __init__(
        self,
        resnet_type: RESNET_T,
        pipeline_stage: STAGE_T,
        model_type: MODEL_T,
        n_prototypes: int | None = None,
        proj_dim: int | None = None,
        label_names: list[str] | None = None,
    ):
        super().__init__()
        self.lr = None
        self.save_hyperparameters()

        # fmt: off
        if model_type == "ResNetClassifier" and pipeline_stage != "train-classifier":
            raise ValueError("Can only use model_type=ResNetClassifier with pipeline_stage=train-classifier")

        if model_type == "PrototypeContraster" and pipeline_stage not in {"learn-prototypes", "project-prototypes"}:
            raise ValueError("Can only use model_type=PrototypeContraster with pipeline_stage=learn-prototypes,project-prototypes")
        # fmt: on

        if model_type == "PrototypeContraster":
            assert n_prototypes is not None
            assert proj_dim is not None
            self.model = PrototypeContraster(
                resnet_type=resnet_type,
                n_prototypes=n_prototypes,
                proj_dim=proj_dim,
            )
        elif model_type == "PrototypeClassifier":
            assert n_prototypes is not None
            assert label_names is not None
            self.model = PrototypeClassifier(
                resnet_type=resnet_type,
                n_prototypes=n_prototypes,
                n_binary_labels=len(label_names),
            )
        elif model_type == "ResNetClassifier":
            assert label_names is not None
            self.model = ResNetClassifier(
                resnet_type=resnet_type,
                n_binary_labels=len(label_names),
            )
        else:
            raise ValueError(f"Unknown model_type {model_type}")

    def _common_step(
        self,
        *,  # enforce kwargs
        batch: dict[str, torch.Tensor],
        stage: str,
        log: bool = True,
        sync_dist: bool = False,
    ) -> tuple[
        torch.Tensor,  # scalar (composite) loss value
        dict[str, torch.Tensor] | None,  # 1D array of predicted probabilities
    ]:
        batch_size = batch["patient_id"].shape[0]

        pipeline_stage: STAGE_T = self.hparams.pipeline_stage  # type: ignore
        if pipeline_stage == "learn-prototypes":
            # x1/x2 keys
            probs = None
            loss = self.model(
                batch["x1"],
                batch["x2"],
            )
            if log:
                self.log(
                    f"{stage}_loss", loss, batch_size=batch_size, sync_dist=sync_dist
                )
        elif pipeline_stage == "train-classifier":
            # waveform/label keys
            (
                _losses,  # (n_labels,)
                _probs,  # (n_labels, B)
            ) = self.model(
                batch["waveform"],  # (B, 12, 10 * freq)
                batch["label"],  # (B, n_labels)
            )
            losses = dict()
            probs = dict()
            label_names: list[str] = self.hparams.label_names  # type: ignore
            assert label_names is not None
            for i, target_name in enumerate(label_names):
                losses[target_name] = _losses[i]
                probs[target_name] = _probs[i]
            loss = self._log_and_composite_losses(
                stage=stage,
                losses=losses,
                batch_size=batch_size,
                log=log,
                sync_dist=sync_dist,
            )
        else:
            raise ValueError(f"Unknown forward step fr pipeline_stage {pipeline_stage}")
        return loss, probs

    def _log_and_composite_losses(
        self,
        *,  # enforce kwargs
        stage: str,
        losses: dict[str, torch.Tensor],
        batch_size: int,
        log: bool = True,
        sync_dist: bool = False,
    ) -> torch.Tensor:
        for k, v in losses.items():
            if log:
                self.log(f"{stage}_{k}_loss", v, batch_size=batch_size)
        loss = torch.stack(list(losses.values())).mean()
        if log:
            self.log(f"{stage}_loss", loss, batch_size=batch_size, sync_dist=sync_dist)
        return loss

    def training_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch=batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch=batch, stage="val", sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch=batch, stage="test", sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        _, probs = self._common_step(batch=batch, stage="predict", log=False)
        return probs


@rank_zero_only
def makedirs_wrapper(save_dir, latest_link):
    os.makedirs(save_dir)
    if os.path.exists(latest_link):
        os.remove(latest_link)
    rel_dir = save_dir
    rel_dir = rel_dir.replace(os.path.dirname(rel_dir), "")
    rel_dir = rel_dir.lstrip(os.path.sep)
    os.symlink(rel_dir, latest_link)


def next_version(path, prefix="v"):
    path = Path(path)
    if not path.exists():
        return f"{prefix}1-{generate_id()}"

    versions = []
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)-(.+)$")

    for p in path.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                versions.append(int(m.group(1)))

    # the "UUID" suffix is needed in case runs are deleted and versions are reused
    # as wandb does not allow name reusage even after "deletion" (not really deleted?)
    return f"{prefix}{max(versions) + 1 if versions else 1}-{generate_id()}"


class StrictWandbLogger(WandbLogger):
    def __init__(self, *, project: str, name: str, save_dir: str):
        run_dir = os.path.join(save_dir, name)
        version = next_version(run_dir)
        latest_link = os.path.join(run_dir, "latest")
        save_dir = os.path.join(run_dir, version)
        self.best_link = os.path.join(save_dir, "best.ckpt")

        super().__init__(project=project, name=name, version=version, save_dir=save_dir)
        if os.path.exists(self.save_dir):  # type: ignore
            raise FileExistsError(
                "\033[91mREAD THIS ERROR MSG: \033[0m"
                f"Experiment already exists at {self.save_dir}."
                " This logger uses some custom logic to put all logs,"
                " checkpoints, and configs related to an experiment"
                " under one directory. Please delete or rename to retry."
            )
        makedirs_wrapper(self.save_dir, latest_link)

    def after_save_checkpoint(self, checkpoint_callback):
        best_model_path = checkpoint_callback.best_model_path
        best_model_path = best_model_path.replace(os.path.dirname(self.best_link), "")
        best_model_path = best_model_path.lstrip(os.path.sep)
        if os.path.exists(self.best_link):
            os.remove(self.best_link)
        os.symlink(
            best_model_path,
            self.best_link,
        )


class PredictionWriter(BasePredictionWriter):
    def __init__(self):
        super().__init__("epoch")

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
    ):
        target_names: list[str] = pl_module.target_names  # type: ignore
        probs = {k: [] for k in target_names}
        for batch_probs in predictions:
            for k in target_names:
                batch_label_prob: torch.Tensor = batch_probs[k]
                probs[k].append(batch_label_prob.numpy())
        to_save = np.stack([np.concat(v) for v in probs.values()]).T
        log_dir: str = trainer.log_dir  # type: ignore
        suffix = ""
        if trainer.global_rank != 0:
            suffix = f"_rank_{trainer.global_rank}"
        np.save(os.path.join(log_dir, f"probs{suffix}.npy"), to_save)


class LitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--pipeline-stage",
            choices=get_args(STAGE_T),
            required=True,
        )
        parser.link_arguments("pipeline_stage", "data.init_args.pipeline_stage")
        parser.link_arguments("pipeline_stage", "model.init_args.pipeline_stage")
        parser.link_arguments(
            "data.label_names",
            "model.init_args.label_names",
            apply_on="instantiate",
        )


def run():
    cli = LitCLI(run=False)
    # cli.trainer.fit(
    #     model=cli.model,
    #     datamodule=cli.datamodule,
    # )
    # cli.trainer.predict(
    #     model=cli.model,
    #     datamodule=cli.datamodule,
    #     ckpt_path=os.path.join(cli.trainer.log_dir, "best.ckpt"),  # type: ignore
    # )


if __name__ == "__main__":
    run()
