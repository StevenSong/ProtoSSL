import torch
import torch.nn as nn
import torch.nn.functional as F

from pass_pclr.defines import RESNET_T
from pass_pclr.models.encoders import BaseEncoder, ResNet1D


class PrototypeEncoder(BaseEncoder):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        n_prototypes: int,
    ):
        self.resnet = ResNet1D(resnet_type=resnet_type)
        self.prototypes = nn.Parameter(
            torch.ones(n_prototypes, self.resnet.emb_dim),
            requires_grad=True,
        )
        self.emb_dim = n_prototypes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = F.cosine_similarity(x, self.prototypes)
        return x  # (B, P)
