from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...defines import RESNET_T
from ._base_encoder import BaseEncoder
from ._resnet import ResNet1D


def pairwise_cosine_similarity(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return F.normalize(X, dim=1) @ F.normalize(Y, dim=1).T


class PrototypeEncoder(BaseEncoder):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        n_prototypes: int,
    ):
        super().__init__()
        self.resnet = ResNet1D(resnet_type=resnet_type)
        self.prototypes = nn.Parameter(
            torch.ones(n_prototypes, self.resnet.emb_dim),
            requires_grad=True,
        )
        self.emb_dim = n_prototypes
        self.sim_fn = pairwise_cosine_similarity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)  # (B, E)
        x = self.sim_fn(x, self.prototypes)
        return x  # (B, P)
