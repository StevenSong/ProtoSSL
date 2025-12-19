import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet
from wandb.util import generate_id

torch.set_float32_matmul_precision("medium")

SPLIT_T = Literal["train", "val", "test"]
RESNET_T = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
CONV_T = Literal["1D", "2D"]

# ===================================================================
# 1D ResNet implementation from bbj-lab/protoecgnet:/src/backbones.py
# ===================================================================


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=12):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


RESNET1D_MODEL_FNS = {
    "resnet18": lambda num_classes: ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes),
    "resnet34": lambda num_classes: ResNet1D(BasicBlock1D, [3, 4, 6, 3], num_classes),
    "resnet50": lambda num_classes: ResNet1D(Bottleneck1D, [3, 4, 6, 3], num_classes),
    "resnet101": lambda num_classes: ResNet1D(Bottleneck1D, [3, 4, 23, 3], num_classes),
    "resnet152": lambda num_classes: ResNet1D(Bottleneck1D, [3, 8, 36, 3], num_classes),
}


def make_resnet1d(
    *,  # enforce kwargs
    resnet_type: RESNET_T,
    num_classes: int,
) -> ResNet1D:
    model_fn = RESNET1D_MODEL_FNS[resnet_type]
    model = model_fn(num_classes=num_classes)

    return model


def make_resnet2d(
    *,  # enforce kwargs
    resnet_type: RESNET_T,
    num_classes: int,
) -> resnet.ResNet:
    model_fn = getattr(resnet, resnet_type)
    model = model_fn(num_classes=num_classes)

    # minimal change to use 1 input channel instead of 3
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    return model


class EchoNextECGDataset(Dataset):
    def __init__(
        self,
        *,  # enforce kwargs
        target_config: str,
        echonext_data: str,
        split: SPLIT_T,
    ):
        with open(target_config, "r") as f:
            config = yaml.safe_load(f)
            mapping = config["target_columns"]  # name --> col
            mapping = {v: k for k, v in mapping.items()}  # col --> name
            target_cols = list(mapping.values())

        echonext_path = Path(echonext_data)
        df = pd.read_csv(echonext_path / "EchoNext_metadata_100k.csv")
        df = df.rename(columns=mapping)
        df = df.loc[df["split"] == split, target_cols].reset_index(drop=True)

        labels = df.to_numpy()  # (N, num_classes)
        self.targets = np.stack([1 - labels, labels], axis=-1)  # (N, num_classes, 2)
        self.waveforms = np.load(echonext_path / f"EchoNext_{split}_waveforms.npy")
        self.tabulars = np.load(
            echonext_path / f"EchoNext_{split}_tabular_features.npy"
        )
        assert len(self.targets) == len(
            self.waveforms
        ), "Targets and Waveforms number of samples differ"
        assert len(self.targets) == len(
            self.tabulars
        ), "Targets and Tabulars number of samples differ"

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        waveform = torch.as_tensor(self.waveforms[i])  # (1, 2500, 12)
        waveform = waveform.mT  # (1, 12, 2500)
        label = torch.as_tensor(self.targets[i])  # (num_classes, 2)

        return {
            "waveform": waveform.to(torch.float32),
            "label": label.to(torch.float32),
        }

    def __len__(self) -> int:
        return len(self.targets)


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


def run():
    cli = LightningCLI(run=False)
    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
    )
    cli.trainer.predict(
        model=cli.model,
        datamodule=cli.datamodule,
        ckpt_path=os.path.join(cli.trainer.log_dir, "best.ckpt"),  # type: ignore
    )


if __name__ == "__main__":
    run()
