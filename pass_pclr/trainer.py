import os
import re
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader
from wandb.util import generate_id

from .datasets import PCLRWrapperDataset, infer_dataset_class_from_path
from .defines import CONV_T, PROT_T, RESNET_T, STAGE_T
from .models import (
    BaseClassifier,
    PrototypeClassifier,
    PrototypeContraster,
    ResNetClassifier,
)

torch.set_float32_matmul_precision("medium")


class LitData(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        pipeline_stage: STAGE_T,
        batch_size: int,
        num_workers: int,
        sampling_rate: int = 100,
        prefetch_factor: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # label names is linked to LitModel via LitCLI
        self.ds_cls, self.label_names = infer_dataset_class_from_path(dataset_path)

    def setup(self, stage: str):
        dataset_path: str = self.hparams.dataset_path  # type: ignore
        sampling_rate: int = self.hparams.sampling_rate  # type: ignore
        pipeline_stage: STAGE_T = self.hparams.pipeline_stage  # type: ignore
        wrap_pclr = pipeline_stage == "learn-prototypes"

        if stage == "fit":
            train_ds = self.ds_cls(
                dataset_path=dataset_path,
                split="train",
                sampling_rate=sampling_rate,
            )
            if wrap_pclr:
                train_ds = PCLRWrapperDataset(train_ds)
            self.train_ds = train_ds
        if stage in ["fit", "validate"]:
            val_ds = self.ds_cls(
                dataset_path=dataset_path,
                split="val",
                sampling_rate=sampling_rate,
            )
            if wrap_pclr:
                val_ds = PCLRWrapperDataset(val_ds)
            self.val_ds = val_ds
        if stage in ["test", "predict"]:
            split = "test"
            if stage == "predict" and pipeline_stage == "project-prototypes":
                # hijack predict for prototype projection over training samples
                split = "train"
                print("======================LitData======================")
                print(
                    "Using training split for prediction data loader for prototype projection stage"
                )
                print("===================================================")

            test_ds = self.ds_cls(
                dataset_path=dataset_path,
                split=split,
                sampling_rate=sampling_rate,
            )
            if wrap_pclr:
                raise ValueError("Should not use PCLR dataset with test/predict stage")
            self.test_ds = test_ds

    def train_dataloader(self):
        pipeline_stage: STAGE_T = self.hparams.pipeline_stage  # type: ignore

        return DataLoader(
            self.train_ds,
            shuffle=pipeline_stage
            not in {
                "project-prototypes",
                "compute-embeddings",
            },  # no shuffle if projecting or embedding
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.hparams.num_workers,  # type: ignore
            prefetch_factor=self.hparams.prefetch_factor,  # type: ignore
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.hparams.num_workers,  # type: ignore
            prefetch_factor=self.hparams.prefetch_factor,  # type: ignore
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.hparams.num_workers,  # type: ignore
            prefetch_factor=self.hparams.prefetch_factor,  # type: ignore
        )

    def predict_dataloader(self):
        return self.test_dataloader()


def warn_unused(**kwargs):
    to_warn = []
    for k, v in kwargs.items():
        if v is not None:
            to_warn.append(k)
    if len(to_warn) != 0:
        print("=================UNUSED_PARAMETERS=================")
        print("WARNING: unused parameters detected")
        for k in to_warn:
            print(f"WARNING: {k} is set but unused")
        print("===================================================")


class LitModel(LightningModule):
    def __init__(
        self,
        resnet_type: RESNET_T,
        conv_type: CONV_T,
        pipeline_stage: STAGE_T,
        prototype_type: PROT_T | None = None,
        n_prototypes: int | None = None,
        label_names: list[str] | None = None,
        pretrained_weights: str | None = None,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
    ):
        super().__init__()
        self.lr = None
        self.save_hyperparameters()

        if pipeline_stage == "learn-prototypes":
            if n_prototypes is None or prototype_type is None:
                raise ValueError(
                    "pipeline_stage=learn-prototypes must be used with model_type=PrototypeContraster and setting n_prototypes AND prototype_type"
                )
            warn_unused(label_names=label_names)
            self.model = PrototypeContraster(
                resnet_type=resnet_type,
                conv_type=conv_type,
                prototype_type=prototype_type,
                n_prototypes=n_prototypes,
                pretrained_weights=pretrained_weights,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
            )
        elif (
            pipeline_stage == "project-prototypes"
            or pipeline_stage == "compute-embeddings"
        ):
            if (
                n_prototypes is None
                or pretrained_weights is None
                or prototype_type is None
            ):
                raise ValueError(
                    "pipeline_stage=[project-prototypes|compute-embeddings] must be used with model_type=PrototypeContraster and setting n_prototypes, pretrained_weights, AND prototype_type"
                )
            warn_unused(label_names=label_names)
            self.model = PrototypeContraster(
                resnet_type=resnet_type,
                conv_type=conv_type,
                prototype_type=prototype_type,
                n_prototypes=n_prototypes,
                pretrained_weights=pretrained_weights,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
            )
        elif pipeline_stage == "train-classifier":
            if (
                n_prototypes is not None
                and label_names is not None
                and prototype_type is not None
            ):
                self.model = PrototypeClassifier(
                    resnet_type=resnet_type,
                    conv_type=conv_type,
                    prototype_type=prototype_type,
                    n_prototypes=n_prototypes,
                    n_binary_labels=len(label_names),
                    pretrained_weights=pretrained_weights,
                    partial_len=partial_len,
                    partial_overlap=partial_overlap,
                )
            elif n_prototypes is None and label_names is not None:
                self.model = ResNetClassifier(
                    resnet_type=resnet_type,
                    conv_type=conv_type,
                    n_binary_labels=len(label_names),
                    pretrained_weights=pretrained_weights,
                )
            else:
                raise ValueError(
                    f"Cannot infer classifier model type from given parameters for pipeline_stage=train-classifier"
                )
        else:
            raise ValueError(f"Unknown pipeline_stage {pipeline_stage}")

        if pretrained_weights is not None and isinstance(self.model, BaseClassifier):
            self.model.freeze_encoder()

        if pipeline_stage == "project-prototypes":
            assert n_prototypes is not None
            # placeholder tensor values for the prediction writer to access, mostly to help with type checking
            self.prototype_sims = [torch.as_tensor(-np.inf)] * n_prototypes
            self.prototype_embs = [torch.empty(0)] * n_prototypes
            self.prototype_ids = [
                (
                    torch.as_tensor(-1),  # patient_id
                    torch.as_tensor(-1),  # ecg_id
                    torch.as_tensor(-1),  # chunk_idx
                )
            ] * n_prototypes

    def _common_step(
        self,
        *,  # enforce kwargs
        batch: dict[str, torch.Tensor],
        stage: str,
        log: bool = True,
        sync_dist: bool = False,
    ) -> tuple[
        torch.Tensor | None,  # scalar (composite) loss value
        dict[str, torch.Tensor]  # 1D arrays of predicted probabilities
        | torch.Tensor  # embeddings
        | None,
    ]:
        batch_size = batch["patient_id"].shape[0]

        pipeline_stage: STAGE_T = self.hparams.pipeline_stage  # type: ignore
        if pipeline_stage == "learn-prototypes":
            assert isinstance(self.model, PrototypeContraster)
            if stage not in ["train", "val"]:
                raise ValueError(
                    f"Cannot use _common_step with pipeline_stage=learn-prototype and (lightning) stage={stage}"
                )
            # x1/x2 keys
            preds = None
            loss = self.model(
                batch["x1"],
                batch["x2"],
            )
            if log:
                self.log(
                    f"{stage}_loss", loss, batch_size=batch_size, sync_dist=sync_dist
                )
        elif pipeline_stage == "project-prototypes":
            assert isinstance(self.model, PrototypeContraster)
            if stage != "predict":
                raise ValueError(
                    f"Cannot use pipeline_stage=project-prototypes with non-predict stage (got stage={stage}).\n"
                    f"We hijack lightning.Trainer.predict with the train set for prototype projection (model in eval mode and write projection metadata)."
                )
            loss, preds = None, None
            # waveform/label keys
            sims = self.model.encoder(batch["waveform"])  # (B, n_prototypes)
            (
                embs,  # (B, chunks, d_emb)
                chunks,  # (B, n_prototypes) - which chunks resulted in the prototype sims?
            ) = self.model.encoder.get_last_embs_and_chunks()
            for prot_idx, curr_sim in enumerate(self.prototype_sims):
                prot_sims = sims[:, prot_idx]
                candidates = (prot_sims > curr_sim).argwhere().squeeze(1)
                if candidates.shape[0] == 0:
                    # none in batch are more similar to any of the prototypes
                    continue
                batch_idx = candidates[0]
                chunk_idx = chunks[batch_idx, prot_idx]
                _id = (
                    batch["patient_id"][batch_idx],
                    batch["ecg_id"][batch_idx],
                    chunk_idx,
                )
                _emb = embs[batch_idx, chunk_idx]
                _sim = prot_sims[batch_idx]

                self.prototype_sims[prot_idx] = _sim
                self.prototype_embs[prot_idx] = _emb
                self.prototype_ids[prot_idx] = _id
        elif pipeline_stage == "compute-embeddings":
            assert isinstance(self.model, PrototypeContraster)
            if stage != "predict":
                raise ValueError(
                    f"Cannot use pipeline_stage=compute-embeddings with non-predict stage (got stage={stage}).\n"
                )
            loss = None
            preds = self.model.encoder(batch["waveform"])  # (B, n_prototypes)
        elif pipeline_stage == "train-classifier":
            assert isinstance(self.model, BaseClassifier)
            # waveform/label keys
            (
                _losses,  # (n_labels,)
                _preds,  # (n_labels, B)
            ) = self.model(
                batch["waveform"],  # (B, 12, 10 * freq)
                batch["label"],  # (B, n_labels)
            )
            losses = dict()
            preds = dict()
            label_names: list[str] = self.hparams.label_names  # type: ignore
            assert label_names is not None
            for i, target_name in enumerate(label_names):
                losses[target_name] = _losses[i]
                preds[target_name] = _preds[i]

            loss = self._log_and_composite_losses(
                stage=stage,
                losses=losses,
                batch_size=batch_size,
                log=log,
                sync_dist=sync_dist,
            )
        else:
            raise ValueError(
                f"Unknown forward step for pipeline_stage {pipeline_stage}"
            )
        return loss, preds

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
        _, preds = self._common_step(batch=batch, stage="predict", log=False)
        return preds

    def on_predict_end(self):
        # if we've hijacked predict to do prototype projection, need to set
        # the prototypes to the projected samples - metadata is saved by
        # prediction writer utilities and checkpoint is saved in main function
        pipeline_stage: STAGE_T = self.hparams.pipeline_stage  # type: ignore
        if pipeline_stage == "project-prototypes":
            assert isinstance(self.model, PrototypeContraster)
            # save prototypes to model parameter
            with torch.no_grad():
                self.model.encoder.prototypes.copy_(torch.stack(self.prototype_embs))


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
    def __init__(
        self,
        *,
        project: str,
        name: str,
        save_dir: str,
        pipeline_stage: STAGE_T,
    ):
        run_dir = os.path.join(save_dir, name, pipeline_stage)
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
        # hard to set pipeline_stage on prediction_writer with CLI link_arguments but we can borrow from lit module
        pipeline_stage: STAGE_T = pl_module.hparams.pipeline_stage  # type: ignore
        if pipeline_stage == "project-prototypes":
            assert hasattr(pl_module, "prototype_ids")
            # predictions are all None (see LitModel above)
            # pl_module has prototype projection metadata to save
            n_prototypes = len(pl_module.prototype_sims)  # type: ignore
            pids = [pid for pid, _, _ in pl_module.prototype_ids]  # type: ignore
            eids = [eid for _, eid, _ in pl_module.prototype_ids]  # type: ignore
            cids = [cid for _, _, cid in pl_module.prototype_ids]  # type: ignore
            meta = pd.DataFrame.from_dict(
                {
                    "prototype_id": np.arange(n_prototypes),
                    "patient_id": torch.stack(pids).tolist(),
                    "ecg_id": torch.stack(eids).tolist(),
                    "chunk_idx": torch.stack(cids).tolist(),
                    "emb_sim": torch.stack(pl_module.prototype_sims).tolist(),  # type: ignore
                },
                orient="columns",
            )
            meta.to_csv(
                os.path.join(trainer.log_dir, "projection_metadata.csv"),  # type: ignore
                index=False,
            )
        elif pipeline_stage == "train-classifier":
            assert isinstance(predictions[0], dict)  # classifiction probabilities
            target_names: list[str] = pl_module.hparams.label_names  # type: ignore
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
        elif pipeline_stage == "compute-embeddings":
            assert (
                hasattr(pl_module, "prediction_split")
                and pl_module.prediction_split is not None
            )
            split = pl_module.prediction_split
            assert isinstance(predictions[0], torch.Tensor)  # embeddings
            embeds = torch.concat(predictions).numpy()  # type: ignore
            log_dir: str = trainer.log_dir  # type: ignore
            np.save(os.path.join(log_dir, f"{split}_embeds.npy"), embeds)
        else:
            raise ValueError(
                f"Unknown how to handle predictions for pipeline_stage={pipeline_stage}"
            )


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
            "pipeline_stage",
            "trainer.logger.init_args.pipeline_stage",
        )
        parser.link_arguments(
            "data.label_names",
            "model.init_args.label_names",
            apply_on="instantiate",
        )


def run():
    cli = LitCLI(run=False)
    pipeline_stage: STAGE_T = cli.config.pipeline_stage

    # NOTE all model/data validation should happen in their respective modules above
    # Assume at this point, we have the correct models/datasets for the given stage
    if pipeline_stage == "learn-prototypes":
        cli.trainer.fit(
            model=cli.model,
            datamodule=cli.datamodule,
        )
    elif pipeline_stage == "project-prototypes":
        # hijack predict over train set for prototype projection (see LitData above)
        cli.trainer.predict(
            model=cli.model,
            datamodule=cli.datamodule,
        )
        ckpt_path = os.path.join(cli.trainer.log_dir, "proj.ckpt")  # type: ignore
        cli.trainer.save_checkpoint(ckpt_path, weights_only=False)
    elif pipeline_stage == "compute-embeddings":
        # hack to pass split name to prediction writer, not used anywhere else
        cli.model.prediction_split = "train"
        cli.datamodule.setup("fit")
        cli.trainer.predict(
            model=cli.model,
            dataloaders=cli.datamodule.train_dataloader(),
        )
        cli.model.prediction_split = "test"
        cli.datamodule.setup("test")
        cli.trainer.predict(
            model=cli.model,
            dataloaders=cli.datamodule.test_dataloader(),
        )
    elif pipeline_stage == "train-classifier":
        cli.trainer.fit(
            model=cli.model,
            datamodule=cli.datamodule,
        )
        cli.trainer.predict(
            model=cli.model,
            datamodule=cli.datamodule,
            ckpt_path=os.path.join(cli.trainer.log_dir, "best.ckpt"),  # type: ignore
        )
    else:
        raise ValueError(f"Unknown pipeline stage {pipeline_stage}")


if __name__ == "__main__":
    run()
