from typing import get_args

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..defines import BACKBONE_T, CONTRASTIVE_T, CONV_T, PROT_T
from ._pretrained_utils import PretrainedMixin
from .encoders import PrototypeEncoder


class PrototypeContraster(PretrainedMixin, nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        contrastive_pair_mode: CONTRASTIVE_T,
        backbone_type: BACKBONE_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes: int,
        proj_dim: int | None = None,
        init_log_temp: float = 0.07,
        learnable_temp: bool = True,
        pretrained_weights: str | None = None,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        prototype_h: int | None = None,
        prototype_w: int | None = None,
        # specified via model_kawrgs in trainer
        do_softmax: bool = False,
        do_weighted_sum: bool = False,
        cola_loss_weight: float = 1.0,
        clar_loss_weight: float = 1.0,
        koleo_loss_weight: float = 1.0,
    ):
        super().__init__()

        if contrastive_pair_mode not in get_args(CONTRASTIVE_T):
            raise ValueError(f"Unknown contrastive_pair_mode={contrastive_pair_mode}.")
        if any(
            [x < 0 for x in [cola_loss_weight, clar_loss_weight, koleo_loss_weight]]
        ):
            raise ValueError("loss weights must be nonnegative.")
        if contrastive_pair_mode == "cola+clar" and (
            cola_loss_weight + clar_loss_weight <= 0
        ):
            raise ValueError("loss weights for cola+clar must be > 0")

        self.encoder = PrototypeEncoder(
            backbone_type=backbone_type,
            n_prototypes=n_prototypes,
            conv_type=conv_type,
            prototype_type=prototype_type,
            input_channels=input_channels,
            partial_len=partial_len,
            partial_overlap=partial_overlap,
            prototype_h=prototype_h,
            prototype_w=prototype_w,
        )
        self.do_softmax = do_softmax
        self.do_weighted_sum = do_weighted_sum
        if do_weighted_sum:
            # use similarity scores to compute weighted sum of prototypes
            # prototypes are h-dimensional
            emb_dim = self.encoder.prototypes.shape[1]
        else:
            # inputting similarity scores directly into projection
            # each similarity vector is length n_prototypes
            emb_dim = self.encoder.prototypes.shape[0]
        if proj_dim is None:
            proj_dim = emb_dim // 2
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, proj_dim),
        )

        self.log_temperature = nn.Parameter(
            torch.ones([]) * np.log(1 / init_log_temp),
            requires_grad=learnable_temp,
        )
        self.contrastive_pair_mode = contrastive_pair_mode
        self.cola_loss_weight = cola_loss_weight
        self.clar_loss_weight = clar_loss_weight
        self.koleo_loss_weight = koleo_loss_weight
        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def _project_pair(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x1.shape == x2.shape

        # prototype similarities: shape (B, P)
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        if self.do_softmax:
            # convert scores to probabilities
            x1 = F.softmax(x1, dim=1)
            x2 = F.softmax(x2, dim=1)

        if self.do_weighted_sum:
            # compute weighted prototypes
            x1 = x1 @ self.encoder.prototypes  # (B, E), E = emb_dim
            x2 = x2 @ self.encoder.prototypes  # (B, E)

        x1 = self.proj(x1)
        x2 = self.proj(x2)

        return x1, x2

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_clar: torch.Tensor | None = None,
        x2_clar: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.contrastive_pair_mode in {"pclr", "cola", "clar"}:
            z1, z2 = self._project_pair(x1, x2)
            simclr_loss = self._simclr_loss(z1, z2)
        elif self.contrastive_pair_mode == "cola+clar":
            if x1_clar is None or x2_clar is None:
                raise ValueError(
                    "contrastive_pair_mode='cola+clar' requires x1_clar and x2_clar."
                )

            z1_cola, z2_cola = self._project_pair(x1, x2)
            loss_cola = self._simclr_loss(z1_cola, z2_cola)

            z1_clar_proj, z2_clar_proj = self._project_pair(x1_clar, x2_clar)
            loss_clar = self._simclr_loss(z1_clar_proj, z2_clar_proj)

            total_weight = self.cola_loss_weight + self.clar_loss_weight
            if total_weight <= 0:
                raise ValueError(
                    "For 'cola+clar', cola_loss_weight + clar_loss_weight must be > 0."
                )

            simclr_loss = (
                self.cola_loss_weight * loss_cola + self.clar_loss_weight * loss_clar
            ) / total_weight

        else:
            raise ValueError(
                f"Unknown contrastive_pair_mode {self.contrastive_pair_mode}"
            )

        koleo_loss = self._koleo_loss(self.encoder.prototypes)
        return {
            "SimCLR": simclr_loss,
            "KoLeo": koleo_loss * self.koleo_loss_weight,
        }

    def _simclr_loss(self, x1: torch.Tensor, x2: torch.Tensor):
        # simclr training objective from: https://arxiv.org/pdf/2002.05709
        # implementation adapted from: https://github.com/google-research/simclr/blob/master/objective.py
        # learnable logit_scale adapted from CLIP: https://arxiv.org/pdf/2103.00020

        # learnable positive temperature
        # TODO may need to do clamping?
        logit_scale = self.log_temperature.exp()

        # normalize projections
        x1 = F.normalize(x1, p=2, dim=1)  # (B, H)
        x2 = F.normalize(x2, p=2, dim=1)  # (B, H)
        B = x1.shape[0]

        # in-batch similarities (should be minimized)
        logits_11 = (x1 @ x1.T) * logit_scale
        logits_22 = (x2 @ x2.T) * logit_scale

        # mask out autosimilarity diagonal
        mask = torch.eye(B, dtype=x1.dtype, device=x1.device)
        logits_11 = (1 - mask) * logits_11 + mask * -100  # CE loss ignore_index
        logits_22 = (1 - mask) * logits_22 + mask * -100

        # cross-batch similarities (pairs should be maximized)
        logits_12 = (x1 @ x2.T) * logit_scale
        logits_21 = (x2 @ x1.T) * logit_scale

        # CE loss over concatenated matrices [A B]
        # where A is the cross-batch similiarities whose diagonal should be maximized (given by target)
        # and B is the in-batch similarities which should be minimized
        labels = torch.arange(B, dtype=torch.long, device=x1.device)
        loss_12 = F.cross_entropy(
            input=torch.concat((logits_12, logits_11), dim=1),
            target=labels,
        )
        loss_21 = F.cross_entropy(
            input=torch.concat((logits_21, logits_22), dim=1),
            target=labels,
        )

        return loss_12 + loss_21

    def _koleo_loss(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # koleo loss implementation from dinov2
        # https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py

        x = F.normalize(x, eps=eps, p=2, dim=-1)  # (B, H)

        # compute pairwise nearest neighbor given L2-normalized vectors
        dots = x @ x.T  # (B, B)

        # trick to fill diagonal with -1
        dots.view(-1)[:: (x.shape[0] + 1)].fill_(-1)

        # get indices of max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # (B,)

        # maximize distance between nearest neighbors
        distances = F.pairwise_distance(
            x,  # (B, H)
            x[I],  # (B, H)
            eps=eps,
            p=2,
        )  # (B,)
        loss = -torch.log(distances + eps).mean()
        return loss

    @property
    def allow_extra_keys(self) -> list[str]:
        return []

    @property
    def allow_missing_keys(self) -> list[str]:
        return []
