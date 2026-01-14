from typing import Callable

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity as cos_sim

from ...defines import RESNET_T
from ._base_encoder import BaseEncoder
from ._resnet import ResNet1D


class PrototypeEncoder(BaseEncoder):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        n_prototypes: int,
        sim_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cos_sim,
    ):
        super().__init__()
        self.resnet = ResNet1D(resnet_type=resnet_type)
        self.prototypes = nn.Parameter(
            torch.ones(n_prototypes, self.resnet.emb_dim),
            requires_grad=True,
        )
        self.emb_dim = n_prototypes
        self.sim_fn = sim_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)  # (B, E)
        x = self.sim_fn(x, self.prototypes.T)
        return x  # (B, P)
