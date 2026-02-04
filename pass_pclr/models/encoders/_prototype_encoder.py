import torch
import torch.nn as nn
import torch.nn.functional as F

from ...defines import CONV_T, PROT_T, RESNET_T
from ._base_encoder import BaseEncoder
from ._resnet1d import ResNet1D
from ._resnet2d import ResNet2D


def pairwise_cosine_similarity(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return F.normalize(X, dim=1) @ F.normalize(Y, dim=1).T


class PrototypeEncoder(BaseEncoder):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        conv_type: CONV_T,
        prototytpe_type: PROT_T,
        n_prototypes: int,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
    ):
        super().__init__()
        if conv_type == "1D":
            self.resnet = ResNet1D(resnet_type=resnet_type)
        elif conv_type == "2D":
            self.resnet = ResNet2D(resnet_type=resnet_type)
        else:
            raise ValueError(f"Unknown conv_type={conv_type}")
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, self.resnet.emb_dim),
            requires_grad=True,
        )
        self.emb_dim = n_prototypes
        self.sim_fn = pairwise_cosine_similarity
        self.prototype_type = prototytpe_type
        if prototytpe_type == "partial":
            assert (
                partial_len is not None
            ), f"Must set partial_len if using prototype_type='partial'"
            assert (
                partial_overlap is not None
            ), f"Must set partial_overlap if using prototype_type='partial'"
            self.partial_len = partial_len
            self.partial_overlap = partial_overlap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prototype_type == "global":
            x = self.resnet(x)  # (B, E)
            x = self.sim_fn(x, self.prototypes)  # (B, P)
            return x
        if self.prototype_type == "partial":
            # x has shape (B, L, T), we will chunk along T axis according to partial_len/overlap
            step = int(self.partial_len * (1 - self.partial_overlap))
            x = x.unfold(2, self.partial_len, step)  # (B, L, num_chunks, partial_len)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, num_chunks, L, partial_len)
            B, N, L, P = x.shape
            x = x.view(B * N, L, P)  # (B*num_chunks, L, partial_len)
            x = self.resnet(x)  # (B*num_chunks, E)
            x = self.sim_fn(x, self.prototypes)  # (B*num_chunks, P)
            x = x.view(B, N, -1)  # (B, num_chunks, P)
            x, _ = x.max(1)  # (B, P)
            return x
        else:
            raise ValueError(f"Unknown prototype_type={self.prototype_type}")
