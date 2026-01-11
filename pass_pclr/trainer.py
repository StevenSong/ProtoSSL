import os
import re
from pathlib import Path
from typing import get_args

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader
from wandb.util import generate_id

from pass_pclr.defines import RESNET_T, STAGE_T
from pass_pclr.models import PrototypeClassifier, PrototypeContraster, ResNetClassifier

torch.set_float32_matmul_precision("medium")


class LitData(LightningDataModule):
    def __init__(
        self,
        target_config: str,
        echonext_data: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = EchoNextECGDataset(
                target_config=self.hparams.target_config,  # type: ignore
                echonext_data=self.hparams.echonext_data,  # type: ignore
                split="train",
            )
        if stage in ["fit", "validate"]:
            self.val_ds = EchoNextECGDataset(
                target_config=self.hparams.target_config,  # type: ignore
                echonext_data=self.hparams.echonext_data,  # type: ignore
                split="val",
            )
        if stage in ["test", "predict"]:
            self.test_ds = EchoNextECGDataset(
                target_config=self.hparams.target_config,  # type: ignore
                echonext_data=self.hparams.echonext_data,  # type: ignore
                split="test",
            )

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
        target_config: str,
        resnet_type: RESNET_T,
        conv_type: CONV_T,
    ):
        super().__init__()
        self.lr = None
        self.save_hyperparameters()

        with open(self.hparams.target_config, "r") as f:  # type: ignore
            target_names = list(yaml.safe_load(f)["target_columns"].keys())
            self.target_names = [x.replace(" ", "_") for x in target_names]
            self.num_classes = len(self.target_names)

        if self.hparams.conv_type == "1D":  # type: ignore
            resnet_fn = make_resnet1d
        elif self.hparams.conv_type == "2D":  # type: ignore
            resnet_fn = make_resnet2d
        else:
            raise ValueError(f"Unknown conv_type: {self.hparams.conv_type}")  # type: ignore

        self.model = resnet_fn(
            resnet_type=self.hparams.resnet_type,  # type: ignore
            num_classes=self.num_classes * 2,  # binary multilabel
        )

    def _common_step(self, batch) -> tuple[
        dict[str, torch.Tensor],  # scalar loss values
        dict[str, torch.Tensor],  # 1D array of predicted probabilities
        int,  # batch size
    ]:
        x = batch["waveform"]  # (B, 1, 2500, 12)
        y = batch["label"]  # (B, num_classes, 2)
        if self.hparams.conv_type == "1D":  # type: ignore
            x = x.squeeze(1)
        logits = self.model(x)  # (B, 2 * num_classes)
        losses = dict()
        probs = dict()
        for i, target_name in enumerate(self.target_names):
            per_label_logits = logits[:, i * 2 : (i + 1) * 2]  # neg/pos per label
            per_label_loss = F.cross_entropy(
                input=per_label_logits,  # (B, 2)
                target=y[:, i],  # (B, 2)
            )
            losses[target_name] = per_label_loss
            probs[target_name] = F.softmax(per_label_logits, dim=1)[:, 1]  # (B,)
        return losses, probs, len(x)

    def _log_and_composite_losses(
        self,
        stage: str,
        losses: dict[str, torch.Tensor],
        batch_size: int,
        sync_dist: bool = False,
    ) -> torch.Tensor:
        for k, v in losses.items():
            self.log(f"{stage}_{k}_loss", v, batch_size=batch_size)
        loss = torch.stack(list(losses.values())).mean()
        self.log(f"{stage}_loss", loss, batch_size=batch_size, sync_dist=sync_dist)
        return loss

    def training_step(self, batch, batch_idx):
        losses, _, n_batch = self._common_step(batch)
        loss = self._log_and_composite_losses("train", losses, n_batch)
        return loss

    def validation_step(self, batch, batch_idx):
        losses, _, n_batch = self._common_step(batch)
        loss = self._log_and_composite_losses("val", losses, n_batch, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        losses, _, n_batch = self._common_step(batch)
        loss = self._log_and_composite_losses("test", losses, n_batch, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        _, probs, _ = self._common_step(batch)
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
        parser.add_argument("--pipeline-stage", choices=get_args(STAGE_T), require=True)


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
