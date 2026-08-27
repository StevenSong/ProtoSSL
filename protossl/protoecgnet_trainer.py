import os

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from torch.utils.data import DataLoader

from .datasets import infer_dataset_class_from_path, validate_label_subset
from .defines import BACKBONE_T, CONV_T, PROT_T, SIM_MAX, STAGE_T
from .lightning_utils import check_final_link
from .models import PrototypeProjector
from .models._protoecgnet import (
    BRANCHES_T,
    LOSS_WEIGHT_T,
    ProtoECGNet,
    ProtoECGNetFusion,
)

torch.set_float32_matmul_precision("medium")

PROTOECGNET_PIPELINE_STAGES = [
    "learn-prototypes-supervised",
    "project-prototypes-supervised",
    "compute-embeddings",
    "train-classifier",
    "train-fusion-classifier",
]

ProtoECGNetTrainerError = ValueError(
    f"This specialized trainer should only be used for ProtoECGNet-style modeling, "
    f"restricted to the following pipeline_stages: {PROTOECGNET_PIPELINE_STAGES}"
)


class LitData(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        pipeline_stage: STAGE_T,
        batch_size: int,
        num_workers: int,
        sampling_rate: int = 100,
        prefetch_factor: int | None = None,
        label_subset: list[str] | None = None,
        data_kwargs: dict = dict(),
    ):
        super().__init__()
        self.save_hyperparameters()

        if pipeline_stage not in PROTOECGNET_PIPELINE_STAGES:
            raise ProtoECGNetTrainerError

        # label names is linked to LitModel via LitCLI
        self.ds_cls, self.label_names, self.is_audio = infer_dataset_class_from_path(
            dataset_path
        )
        if label_subset is not None:
            if self.label_names is None:
                raise ValueError(
                    f"Can only specify label_subset for a dataset with defined labels"
                )
            validate_label_subset(label_subset, self.label_names)
            self.label_names = label_subset

        if (
            pipeline_stage == "learn-prototypes-supervised"
            or pipeline_stage == "train-classifier"
            or pipeline_stage == "train-fusion-classifier"
        ):
            # need label weights and cooccurrence matrix, not best practice
            # to instantiate dataset outside of `setup` but we know we'll need
            # the train_ds in this pipeline_stage anyways
            self.train_ds = self.ds_cls(
                dataset_path=dataset_path,
                split="train",
                sampling_rate=sampling_rate,
                label_subset=label_subset,
                **data_kwargs,
            )
            self.label_weights = self.train_ds.get_label_weights()
            self.label_cooccurrence = self.train_ds.get_label_cooccurrence()
        else:
            # set to defaults so LitCLI can link these to LitModel args
            self.label_weights = None
            self.label_cooccurrence = None

    def setup(self, stage: str):
        dataset_path: str = self.hparams.dataset_path  # type: ignore
        sampling_rate: int = self.hparams.sampling_rate  # type: ignore
        label_subset: list[str] | None = self.hparams.label_subset  # type: ignore
        data_kwargs: dict = self.hparams.data_kwargs  # type: ignore

        if stage == "fit":
            if not hasattr(self, "train_ds"):
                self.train_ds = self.ds_cls(
                    dataset_path=dataset_path,
                    split="train",
                    sampling_rate=sampling_rate,
                    label_subset=label_subset,
                    **data_kwargs,
                )
            else:
                assert self.train_ds is not None, "Not sure how train_ds is None"

        if stage in ["fit", "validate"]:
            self.val_ds = self.ds_cls(
                dataset_path=dataset_path,
                split="val",
                sampling_rate=sampling_rate,
                label_subset=label_subset,
                **data_kwargs,
            )

        if stage in ["test", "predict"]:
            self.test_ds = self.ds_cls(
                dataset_path=dataset_path,
                split="test",
                sampling_rate=sampling_rate,
                label_subset=label_subset,
                **data_kwargs,
            )

    def train_dataloader(self):
        pipeline_stage: STAGE_T = self.hparams.pipeline_stage  # type: ignore

        return DataLoader(
            self.train_ds,
            shuffle=(
                # no shuffle if projecting/embedding
                pipeline_stage
                not in {
                    "project-prototypes-supervised",
                    "compute-embeddings",
                }
            ),
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


class LitModel(LightningModule):
    def __init__(
        self,
        pipeline_stage: STAGE_T,
        label_names: list[str],
        backbone_type: BACKBONE_T | None = None,
        conv_type: CONV_T | None = None,
        prototype_type: PROT_T | None = None,
        n_prototypes_per_label: int | None = None,
        label_weights: torch.Tensor | None = None,
        label_cooccurrence: torch.Tensor | None = None,
        pretrained_weights: str | None = None,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        loss_weights: LOSS_WEIGHT_T | None = None,
        branches: BRANCHES_T | None = None,
    ):
        super().__init__()
        self.lr = None
        self.save_hyperparameters()

        if pipeline_stage not in PROTOECGNET_PIPELINE_STAGES:
            raise ProtoECGNetTrainerError

        n_prototypes = None
        if pipeline_stage == "learn-prototypes-supervised":
            assert label_weights is not None, "label_weights is None"
            assert label_cooccurrence is not None, "label_cooccurrence is None"
            assert backbone_type is not None, "backbone_type is None"
            assert conv_type is not None, "conv_type is None"
            assert prototype_type is not None, "prototype_type is None"
            assert n_prototypes_per_label is not None, "n_prototypes_per_label is None"
            # don't enforce default at litmodel, let model handle defaults
            model_kwargs = dict()
            if loss_weights is not None:
                model_kwargs["loss_weights"] = loss_weights
            self.model = ProtoECGNet(
                pipeline_stage=pipeline_stage,
                backbone_type=backbone_type,
                conv_type=conv_type,
                prototype_type=prototype_type,
                n_prototypes_per_label=n_prototypes_per_label,
                label_names=label_names,
                label_weights=label_weights,
                label_cooccurrence=label_cooccurrence,
                pretrained_weights=pretrained_weights,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
                **model_kwargs,
            )
        elif (
            pipeline_stage == "project-prototypes-supervised"
            or pipeline_stage == "compute-embeddings"
        ):
            self._batch_counter = 0
            assert pretrained_weights is not None, "pretrained_weights is None"
            assert backbone_type is not None, "backbone_type is None"
            assert conv_type is not None, "conv_type is None"
            assert prototype_type is not None, "prototype_type is None"
            assert n_prototypes_per_label is not None, "n_prototypes_per_label is None"
            n_prototypes = n_prototypes_per_label * len(label_names)
            self.model = PrototypeProjector(
                backbone_type=backbone_type,
                conv_type=conv_type,
                prototype_type=prototype_type,
                n_prototypes=n_prototypes,
                pretrained_weights=pretrained_weights,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
            )
        elif pipeline_stage == "train-classifier":
            assert label_weights is not None, "label_weights is None"
            assert pretrained_weights is not None, "pretrained_weights is None"
            assert backbone_type is not None, "backbone_type is None"
            assert conv_type is not None, "conv_type is None"
            assert prototype_type is not None, "prototype_type is None"
            assert n_prototypes_per_label is not None, "n_prototypes_per_label is None"
            # don't enforce default at litmodel, let model handle defaults
            model_kwargs = dict()
            if loss_weights is not None:
                model_kwargs["loss_weights"] = loss_weights
            self.model = ProtoECGNet(
                pipeline_stage=pipeline_stage,
                backbone_type=backbone_type,
                conv_type=conv_type,
                prototype_type=prototype_type,
                n_prototypes_per_label=n_prototypes_per_label,
                label_names=label_names,
                label_weights=label_weights,
                pretrained_weights=pretrained_weights,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
                **model_kwargs,
            )
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        elif pipeline_stage == "train-fusion-classifier":
            assert label_weights is not None, "label_weights is None"
            assert branches is not None, "branches is None"
            # don't enforce default at litmodel, let model handle defaults
            model_kwargs = dict()
            if loss_weights is not None:
                model_kwargs["loss_weights"] = loss_weights
            self.model = ProtoECGNetFusion(
                pipeline_stage=pipeline_stage,
                label_names=label_names,
                label_weights=label_weights,
                branches=branches,
                pretrained_weights=pretrained_weights,
                **model_kwargs,
            )
            for param in self.model.encoders.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Unknown pipeline_stage {pipeline_stage}")

        if pipeline_stage == "project-prototypes-supervised":
            assert n_prototypes is not None
            # placeholder tensor values for the prediction writer to access, mostly to help with type checking
            self.prototype_sims = torch.ones((n_prototypes,)) * -SIM_MAX
            self.prototype_embs = torch.empty(
                (n_prototypes, self.model.encoder.backbone.emb_dim)  # type: ignore
            )
            # each row is source_id, sample_id, chunk_idx
            self.prototype_ids = torch.empty((n_prototypes, 3), dtype=torch.long)

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
        | tuple[torch.Tensor, torch.Tensor]  # embeddings, metadata
        | None,
    ]:
        batch_size = batch["source_id"].shape[0]
        pipeline_stage: STAGE_T = self.hparams.pipeline_stage  # type: ignore

        if (
            pipeline_stage == "learn-prototypes-supervised"
            or pipeline_stage == "train-classifier"
            or pipeline_stage == "train-fusion-classifier"
        ):
            loss_terms, preds = self.model(batch["waveform"], batch["label"])
            loss = self._log_and_composite_losses(
                stage=stage,
                losses=loss_terms,
                batch_size=batch_size,
                log=log,
                sync_dist=sync_dist,
            )
        elif pipeline_stage == "project-prototypes-supervised":
            self._batch_counter += 1
            assert isinstance(self.model, PrototypeProjector)
            if stage != "predict":
                raise ValueError(
                    f"Cannot use pipeline_stage=project-prototypes with non-predict stage (got stage={stage}).\n"
                    f"We hijack lightning.Trainer.predict with the train set for prototype projection (model in eval mode and write projection metadata)."
                )
            loss, preds = None, None
            # waveform/label keys
            sims = self.model(batch["waveform"])  # (B, n_prototypes)
            if pipeline_stage == "project-prototypes-supervised":
                # mask out prototypes for which the sample cannot belong to
                _, L = batch["label"].shape  # (B, L)
                _, P = sims.shape
                ppl = P // L  # n_prototypes_per_label
                pos_mask = batch["label"].repeat_interleave(ppl, 1)  # (B, P)
                sims = pos_mask * sims + (1 - pos_mask) * -SIM_MAX  # (B, P)
            (
                embs,  # (B, chunks, d_emb)
                chunks,  # (B, n_prototypes) - which chunks resulted in the prototype sims?
            ) = self.model.get_last_embs_and_chunks()

            # move things off gpu
            sims, embs, chunks = (
                sims.detach().cpu(),
                embs.detach().cpu(),
                chunks.detach().cpu(),
            )
            source_ids, sample_ids = batch["source_id"].cpu(), batch["sample_id"].cpu()

            # find max sim
            for prot_idx, curr_sim in enumerate(self.prototype_sims):
                prot_sims = sims[:, prot_idx]
                candidates = (prot_sims > curr_sim).argwhere().squeeze(1)
                if candidates.shape[0] == 0:
                    # none in batch are more similar to any of the prototypes
                    continue
                # get the candidate in batch that is the most similar
                batch_idx = candidates[prot_sims[candidates].argmax()]
                chunk_idx = chunks[batch_idx, prot_idx]

                self.prototype_embs[prot_idx] = embs[batch_idx, chunk_idx]
                self.prototype_sims[prot_idx] = prot_sims[batch_idx]
                self.prototype_ids[prot_idx, 0] = source_ids[batch_idx]
                self.prototype_ids[prot_idx, 1] = sample_ids[batch_idx]
                self.prototype_ids[prot_idx, 2] = chunk_idx
        elif pipeline_stage == "compute-embeddings":
            assert isinstance(self.model, PrototypeProjector)
            if stage != "predict":
                raise ValueError(
                    f"Cannot use pipeline_stage=compute-embeddings with non-predict stage (got stage={stage}).\n"
                )
            loss = None
            sims = self.model(batch["waveform"])  # (B, n_prototypes)
            # which chunks resulted in the prototype sims?
            _, chunks = self.model.get_last_embs_and_chunks()  # (B, n_prototypes)

            preds = sims, chunks
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
                self.log(
                    f"{stage}_{k}_loss", v, batch_size=batch_size, sync_dist=sync_dist
                )
        loss = torch.stack(list(losses.values())).sum()  # NOTE: sum, not mean
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

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        pipeline_stage: STAGE_T = self.hparams.pipeline_stage  # type: ignore
        if pipeline_stage == "project-prototypes-supervised":
            assert isinstance(self.model, PrototypeProjector)

            # sync projected prototypes across distributed ranks
            # NOTE: for speed, we only run this once (in the last batch)
            # it would be cleaner to put this in on_predict_end, however the
            # prediction writer's on_predict_end hook fires before the model's
            # and we shouldn't really rely on enforcing peer callbacks' order
            if (
                len(self.trainer.predict_dataloaders) == batch_idx + 1  # type: ignore
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                world_size = torch.distributed.get_world_size()
                # fmt: off
                gathered_embs = [None for _ in range(world_size)]
                gathered_sims = [None for _ in range(world_size)]
                gathered_ids = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(gathered_embs, self.prototype_embs)
                torch.distributed.all_gather_object(gathered_sims, self.prototype_sims)
                torch.distributed.all_gather_object(gathered_ids, self.prototype_ids)
                gathered_embs = torch.stack(gathered_embs) # type: ignore
                gathered_sims = torch.stack(gathered_sims) # type: ignore
                gathered_ids = torch.stack(gathered_ids) # type: ignore
                torch.distributed.barrier()
                # fmt: on

                _, idxs = gathered_sims.max(dim=0)  # (P,)
                for p in range(len(idxs)):
                    idx = idxs[p]  # for proto p, rank with max sim
                    self.prototype_embs[p] = gathered_embs[idx][p]  # type: ignore
                    self.prototype_sims[p] = gathered_sims[idx][p]  # type: ignore
                    self.prototype_ids[p] = gathered_ids[idx][p]  # type: ignore

    def on_predict_end(self):
        # if we've hijacked predict to do prototype projection, need to set
        # the prototypes to the projected samples - metadata is saved by
        # prediction writer utilities and checkpoint is saved in main function
        pipeline_stage: STAGE_T = self.hparams.pipeline_stage  # type: ignore
        if pipeline_stage == "project-prototypes-supervised":
            assert isinstance(self.model, PrototypeProjector)

            # save projected prototypes to model parameter
            with torch.no_grad():
                self.model.encoder.prototypes.copy_(self.prototype_embs)


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

        if pipeline_stage == "project-prototypes-supervised":
            assert hasattr(pl_module, "prototype_ids")
            # predictions are all None (see LitModel above)
            # pl_module has prototype projection metadata to save
            meta = pd.DataFrame.from_dict(
                {
                    "prototype_id": np.arange(len(pl_module.prototype_sims)),  # type: ignore
                    "source_id": pl_module.prototype_ids[:, 0].tolist(),  # type: ignore
                    "sample_id": pl_module.prototype_ids[:, 1].tolist(),  # type: ignore
                    "chunk_idx": pl_module.prototype_ids[:, 2].tolist(),  # type: ignore
                    "emb_sim": pl_module.prototype_sims.tolist(),  # type: ignore
                },
                orient="columns",
            )
            # after gathering data to save, only save on rank 0
            if trainer.global_rank == 0:
                meta.to_csv(
                    os.path.join(trainer.log_dir, "projection_metadata.csv"),  # type: ignore
                    index=False,
                )
            to_save = None  # type: ignore
            save_name = None
        elif (
            pipeline_stage == "train-classifier"
            or pipeline_stage == "train-fusion-classifier"
        ):
            # TODO batch order when using distributed
            assert isinstance(predictions[0], dict)  # classifiction probabilities
            if "Multiclass" in predictions[0]:
                to_save = np.concatenate([p["Multiclass"] for p in predictions])
            else:
                target_names: list[str] = pl_module.hparams.label_names  # type: ignore
                probs = {k: [] for k in target_names}
                for batch_probs in predictions:
                    for k in target_names:
                        batch_label_prob: torch.Tensor = batch_probs[k]
                        probs[k].append(batch_label_prob.numpy())
                to_save = np.stack([np.concatenate(v) for v in probs.values()]).T
            save_name = "probs.npy"
        elif pipeline_stage == "compute-embeddings":
            assert (
                hasattr(pl_module, "prediction_split")
                and pl_module.prediction_split is not None
            )
            split = pl_module.prediction_split
            assert isinstance(predictions[0], tuple)
            assert isinstance(predictions[0][0], torch.Tensor)  # embeddings
            assert isinstance(predictions[0][1], torch.Tensor)  # chunk metadata
            embeddings = [p[0] for p in predictions]
            to_save = torch.concatenate(embeddings).numpy()  # type: ignore
            save_name = f"{split}_embeds.npy"

            # TODO: gather/save chunk metadata with DDP
            chunk_metadata = [p[1] for p in predictions]
            chunk_metadata = torch.concatenate(chunk_metadata).numpy()
            np.save(os.path.join(trainer.log_dir, f"{split}_chunks.npy"), chunk_metadata)  # type: ignore
        else:
            raise ValueError(
                f"Unknown how to handle predictions for pipeline_stage={pipeline_stage}"
            )

        # nothing to save or save handled separately
        if to_save is None:
            return

        # NOTE: currently only single device and DDP strategies have been validated
        # saving (potentially distributed) predictions

        # batch_indices is a singleton list of a list of lists
        # e.g. [[[0, 2], [1, 3]]], where there are 4 samples and 2 batches of 2 samples each
        assert len(batch_indices) == 1  # do other strategies return different idxs?
        flat_idxs = np.asarray([x for xs in batch_indices[0] for x in xs])

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_to_save = [None] * world_size
            gathered_flat_idxs = [None] * world_size
            torch.distributed.all_gather_object(gathered_to_save, to_save)
            torch.distributed.all_gather_object(gathered_flat_idxs, flat_idxs)
            torch.distributed.barrier()
            to_save: np.ndarray = np.concatenate(gathered_to_save)  # type: ignore
            flat_idxs: np.ndarray = np.concatenate(gathered_flat_idxs)  # type: ignore
            sort_mask = flat_idxs.argsort()
            to_save = to_save[sort_mask]

        # after gathering data to save, only save on rank 0
        if trainer.global_rank != 0:
            return
        log_dir: str = trainer.log_dir  # type: ignore
        np.save(os.path.join(log_dir, save_name), to_save)  # type: ignore


class LitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--pipeline-stage",
            choices=PROTOECGNET_PIPELINE_STAGES,
            required=True,
        )
        parser.add_argument("--resume-from-checkpoint")
        parser.link_arguments(
            "resume_from_checkpoint", "trainer.logger.init_args.resume_from_checkpoint"
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
        parser.link_arguments(
            "data.label_weights",
            "model.init_args.label_weights",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.label_cooccurrence",
            "model.init_args.label_cooccurrence",
            apply_on="instantiate",
        )


def run():
    cli = LitCLI(
        run=False,
        save_config_kwargs={"overwrite": True},  # for resuming checkpoint in same dir
    )
    _seed = os.environ.get("PL_GLOBAL_SEED", None)
    if _seed is not None:
        if int(_seed) < 42:
            # this is a silly limitation tied to our initial results
            # where ILP prototype assignment had random seed = 0 while the rest
            # of the initial results used random seed 42 elsewhere. to ensure
            # reproducibility, we do seed - 42 for ILP assignment
            raise ValueError("seed_everything must be 42 or greater")
    if not isinstance(cli.trainer.strategy, (DDPStrategy, SingleDeviceStrategy)):
        # NOTE: to implement support for other distributed startegies, should check
        # the places noted in this GH issue: https://github.com/StevenSong/ecg-prototype-fm/issues/63
        raise ValueError(
            f"Only single device or DDP training strategies are supported, got: {cli.trainer.strategy}"
        )
    pipeline_stage: STAGE_T = cli.config.pipeline_stage

    # NOTE all model/data validation should happen in their respective modules above
    # Assume at this point, we have the correct models/datasets for the given stage
    if pipeline_stage == "learn-prototypes-supervised":
        cli.trainer.fit(
            model=cli.model,
            datamodule=cli.datamodule,
            ckpt_path=cli.config.resume_from_checkpoint,
        )
    elif pipeline_stage == "project-prototypes-supervised":
        # hijack predict over train set for prototype projection (see LitData above)
        cli.datamodule.setup("fit")
        cli.trainer.predict(
            model=cli.model,
            dataloaders=cli.datamodule.train_dataloader(),
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
    elif (
        pipeline_stage == "train-classifier"
        or pipeline_stage == "train-fusion-classifier"
    ):
        cli.trainer.fit(
            model=cli.model,
            datamodule=cli.datamodule,
            ckpt_path=cli.config.resume_from_checkpoint,
        )
        cli.trainer.predict(
            model=cli.model,
            datamodule=cli.datamodule,
        )
    else:
        raise ValueError(f"Unknown pipeline stage {pipeline_stage}")
    check_final_link(cli.trainer.log_dir)


if __name__ == "__main__":
    run()
