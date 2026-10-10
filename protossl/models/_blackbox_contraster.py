from typing import get_args

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..defines import BACKBONE_T, CONTRASTIVE_T, CONV_T
from ._pretrained_utils import PretrainedMixin
from .encoders import ResNet1D, ResNet2D


class BlackboxContraster(PretrainedMixin, nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        contrastive_pair_mode: CONTRASTIVE_T,
        backbone_type: BACKBONE_T,
        conv_type: CONV_T,
        proj_dim: int | None = None,
        init_log_temp: float = 0.07,
        learnable_temp: bool = True,
        pretrained_weights: str | None = None,
        input_channels: int = 12,
        # specified via model_kawrgs in trainer
    ):
        super().__init__()

        if contrastive_pair_mode not in get_args(CONTRASTIVE_T):
            raise ValueError(f"Unknown contrastive_pair_mode={contrastive_pair_mode}.")
        if contrastive_pair_mode != "pclr":
            raise NotImplementedError(
                f"Non-PCLR contrastive_pair_mode ({contrastive_pair_mode}) not yet supported for BlackboxContraster"
            )
        if not backbone_type.startswith("resnet"):
            raise NotImplementedError(
                f"Non-resnet (lower-cased) backbone_type ({backbone_type}) not yet supported for BlackboxContraster"
            )

        if conv_type == "1D":
            backbone_cls = ResNet1D
        elif conv_type == "2D":
            backbone_cls = ResNet2D
        elif conv_type == "PANNS":
            raise NotImplemented(
                f"PANNS conv_type not yet supported for BlackboxContraster"
            )
        else:
            raise ValueError(f"Unknown conv_type={conv_type}")

        self.encoder = backbone_cls(
            backbone_type=backbone_type,
            input_channels=input_channels,
        )
        emb_dim = self.encoder.emb_dim
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
        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def _project_pair(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x1.shape == x2.shape

        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x1 = self.proj(x1)
        x2 = self.proj(x2)

        return x1, x2

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z1, z2 = self._project_pair(x1, x2)
        simclr_loss = self._simclr_loss(z1, z2)

        return {"SimCLR": simclr_loss}

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

    @property
    def allow_extra_keys(self) -> list[str]:
        return []

    @property
    def allow_missing_keys(self) -> list[str]:
        return []
