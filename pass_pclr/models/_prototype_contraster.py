import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..defines import CONV_T, PROT_T, RESNET_T
from ._pretrained_utils import PretrainedMixin
from .encoders import PrototypeEncoder

class PrototypeContraster(PretrainedMixin, nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
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
        audio_backbone_name: str | None = None,
        contrastive_pair_mode: str = "pclr",
        cola_loss_weight: float = 1.0,
        clar_loss_weight: float = 1.0,
        koleo_loss_weight: float = 1.0,
    ):
        super().__init__()
        valid_modes = {"pclr", "cola", "clar", "cola+clar"}
        if contrastive_pair_mode not in valid_modes:
            raise ValueError(
                f"Unknown contrastive_pair_mode={contrastive_pair_mode}. "
                f"Expected one of {sorted(valid_modes)}"
            )

        if cola_loss_weight < 0 or clar_loss_weight < 0:
            raise ValueError("cola_loss_weight and clar_loss_weight must be nonnegative.")
        if koleo_loss_weight < 0:
            raise ValueError("koleo_loss_weight must be nonnegative.")

        self.encoder = PrototypeEncoder(
            resnet_type=resnet_type,
            n_prototypes=n_prototypes,
            conv_type=conv_type,
            prototype_type=prototype_type,
            input_channels=input_channels,
            partial_len=partial_len,
            partial_overlap=partial_overlap,
            prototype_h=prototype_h,
            prototype_w=prototype_w,
            audio_backbone_name=audio_backbone_name,
        )
        # We now contrast DIRECTLY on prototype activations, so the projection
        # head should take a length-P activation vector as input.
        if proj_dim is None:
            proj_dim = max(128, n_prototypes // 2)

        self.proj = nn.Sequential(
            nn.Linear(n_prototypes, n_prototypes),
            nn.ReLU(inplace=True),
            nn.Linear(n_prototypes, proj_dim),
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

        # Raw pooled prototype activations/similarities: shape (B, P)
        a1 = self.encoder(x1)
        a2 = self.encoder(x2)

        # Contrast directly on prototype activations via a small projection head.
        z1 = self.proj(a1)
        z2 = self.proj(a2)

        return z1, z2

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_clar: torch.Tensor | None = None,
        x2_clar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.contrastive_pair_mode in {"pclr", "cola", "clar"}:
            z1, z2 = self._project_pair(x1, x2)
            simclr_loss = self._simclr_loss(z1, z2)

        elif self.contrastive_pair_mode == "cola+clar":
            if x1_clar is None or x2_clar is None:
                raise ValueError("contrastive_pair_mode='cola+clar' requires x1_clar and x2_clar.")

            z1_cola, z2_cola = self._project_pair(x1, x2)
            loss_cola = self._simclr_loss(z1_cola, z2_cola)

            z1_clar_proj, z2_clar_proj = self._project_pair(x1_clar, x2_clar)
            loss_clar = self._simclr_loss(z1_clar_proj, z2_clar_proj)

            total_weight = self.cola_loss_weight + self.clar_loss_weight
            if total_weight <= 0:
                raise ValueError("For 'cola+clar', cola_loss_weight + clar_loss_weight must be > 0.")

            simclr_loss = (
                self.cola_loss_weight * loss_cola
                + self.clar_loss_weight * loss_clar
            ) / total_weight

        else:
            raise ValueError(f"Unknown contrastive_pair_mode {self.contrastive_pair_mode}")

        koleo_loss = self._koleo_loss(self.encoder.prototypes)
        return simclr_loss + self.koleo_loss_weight * koleo_loss

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
