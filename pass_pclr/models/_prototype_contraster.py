import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pass_pclr.defines import RESNET_T
from pass_pclr.models.encoders import PrototypeEncoder


class PrototypeContraster(nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        n_prototypes: int,
        proj_dim: int,
        init_log_temp: float = 0.07,
        learnable_temp: bool = True,
    ):
        self.prototype_encoder = PrototypeEncoder(
            resnet_type=resnet_type,
            n_prototypes=n_prototypes,
        )
        self.proj = nn.Linear(
            in_features=self.prototype_encoder.prototypes.shape[1],
            out_features=proj_dim,
        )
        self.log_temperature = nn.Parameter(
            torch.ones([]) * np.log(1 / init_log_temp),
            requires_grad=learnable_temp,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        assert x1.shape == x2.shape

        # compute prototype similarities
        x1 = self.prototype_encoder(x1)  # (B, P), P = n_prototypes
        x2 = self.prototype_encoder(x2)  # (B, P)

        # compute weighted prototypes
        x1 = x1 @ self.prototype_encoder.prototypes  # (B, E), E = emb_dim
        x2 = x2 @ self.prototype_encoder.prototypes  # (B, E)

        # projection
        x1 = self.proj(x1)  # (B, H), H = proj_dim
        x2 = self.proj(x2)  # (B, H)

        # simclr loss
        loss = self._simclr_loss(x1, x2)
        return loss

    def _simclr_loss(self, x1: torch.Tensor, x2: torch.Tensor):
        # simclr training objective from SimCLR: https://arxiv.org/pdf/2002.05709
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
        mask = torch.eye(B)
        logits_11 = (1 - mask) * logits_11 + mask * -100  # CE loss ignore_index
        logits_22 = (1 - mask) * logits_22 + mask * -100

        # cross-batch similarities (pairs should be maximized)
        logits_12 = (x1 @ x2.T) * logit_scale
        logits_21 = (x2 @ x1.T) * logit_scale

        # CE loss over concatenated matrices [A B]
        # where A is the cross-batch similiarities whose diagonal should be maximized (given by target)
        # and B is the in-batch similarities which should be minimized
        labels = torch.arange(B)
        loss_12 = F.cross_entropy(
            input=torch.concat((logits_12, logits_11), dim=1),
            target=labels,
        )
        loss_21 = F.cross_entropy(
            input=torch.concat((logits_21, logits_22), dim=1),
            target=labels,
        )

        return loss_12 + loss_21
