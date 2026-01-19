import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..defines import RESNET_T
from ._pretrained_utils import PretrainedMixin
from .encoders import PrototypeEncoder


class PrototypeContraster(PretrainedMixin, nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        n_prototypes: int,
        proj_dim: int | None = None,
        init_log_temp: float = 0.07,
        learnable_temp: bool = True,
        pretrained_weights: str | None = None,
    ):
        super().__init__()
        self.encoder = PrototypeEncoder(
            resnet_type=resnet_type,
            n_prototypes=n_prototypes,
        )
        emb_dim = self.encoder.prototypes.shape[1]
        if proj_dim is None:
            proj_dim = emb_dim // 2
        self.proj = nn.Linear(
            in_features=emb_dim,
            out_features=proj_dim,
        )
        self.log_temperature = nn.Parameter(
            torch.ones([]) * np.log(1 / init_log_temp),
            requires_grad=learnable_temp,
        )
        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        assert x1.shape == x2.shape

        # compute prototype similarity scores
        x1 = self.encoder(x1)  # (B, P), P = n_prototypes
        x2 = self.encoder(x2)  # (B, P)

        # convert scores to probabilities
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)

        # compute weighted prototypes
        x1 = x1 @ self.encoder.prototypes  # (B, E), E = emb_dim
        x2 = x2 @ self.encoder.prototypes  # (B, E)

        # projection
        x1 = self.proj(x1)  # (B, H), H = proj_dim
        x2 = self.proj(x2)  # (B, H)

        # compute losses
        simclr_loss = self._simclr_loss(x1, x2)
        koleo_loss = self._koleo_loss(x1) + self._koleo_loss(x2)
        return simclr_loss + koleo_loss

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
