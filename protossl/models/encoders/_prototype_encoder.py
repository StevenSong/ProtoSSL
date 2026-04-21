import torch
import torch.nn as nn
import torch.nn.functional as F

from ...defines import BACKBONE_T, CONV_T, PROT_T
from ._base_encoder import BaseEncoder
from ._net1d import Net1D
from ._panns_encoder import PANNSEncoder
from ._resnet1d import ResNet1D
from ._resnet2d import ResNet2D


def pairwise_cosine_similarity(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return F.normalize(X, dim=1) @ F.normalize(Y, dim=1).T


def patchwise_cosine_similarity_2d(
    feature_map: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_h: int,
    prototype_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    feature_map: (B, C, H, W)
    prototypes:  (P, D), where D = C * prototype_h * prototype_w

    Returns:
      sims: (B, L, P), where L = number of valid spatial locations
      patch_vectors: (B, L, D)
      grid_hw: (H_out, W_out)
    """
    B, C, H, W = feature_map.shape
    P, D = prototypes.shape
    expected_D = C * prototype_h * prototype_w
    assert D == expected_D, (
        f"Prototype dim mismatch: got D={D}, expected {expected_D} "
        f"from feature map C={C} and patch size ({prototype_h},{prototype_w})"
    )

    patch_vectors = F.unfold(
        feature_map,
        kernel_size=(prototype_h, prototype_w),
        stride=1,
    )  # (B, D, L)
    patch_vectors = patch_vectors.transpose(1, 2).contiguous()  # (B, L, D)

    patch_vectors_n = F.normalize(patch_vectors, dim=-1)
    prototypes_n = F.normalize(prototypes, dim=-1)
    sims = torch.einsum("bld,pd->blp", patch_vectors_n, prototypes_n)  # (B, L, P)

    H_out = H - prototype_h + 1
    W_out = W - prototype_w + 1
    assert (
        H_out > 0 and W_out > 0
    ), f"Prototype size ({prototype_h},{prototype_w}) exceeds feature map size ({H},{W})"

    return sims, patch_vectors


class PrototypeEncoder(BaseEncoder):
    def __init__(
        self,
        *,  # enforce kwargs
        backbone_type: BACKBONE_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes: int,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        prototype_h: int | None = None,
        prototype_w: int | None = None,
    ):
        super().__init__()
        if backbone_type == "net1d":
            if conv_type != "1D":
                raise ValueError("Must use conv_type=1D if backbone_type=net1D")
            backbone_cls = Net1D
        else:
            if conv_type == "1D":
                backbone_cls = ResNet1D
            elif conv_type == "2D":
                backbone_cls = ResNet2D
            elif conv_type == "PANNS":
                backbone_cls = PANNSEncoder
            else:
                raise ValueError(f"Unknown conv_type={conv_type}")
        self.backbone = backbone_cls(
            backbone_type=backbone_type, input_channels=input_channels
        )
        self.conv_type = conv_type
        self.prototype_type = prototype_type
        self.prototype_h = prototype_h
        self.prototype_w = prototype_w
        self.partial_len = partial_len
        self.partial_overlap = partial_overlap

        if self.prototype_type == "global":
            assert (
                conv_type != "PANNS"
            ), "Using PrototypeEncoder with global prototypes not supported with PANNS encoder"
            prot_dim = self.backbone.emb_dim
        elif self.prototype_type == "partial":
            if conv_type == "PANNS":
                assert prototype_h is not None and prototype_w is not None, (
                    "For conv_type='HTSAT' or 'PANNS' and prototype_type='partial', "
                    "set prototype_h and prototype_w explicitly."
                )
                prot_dim = self.backbone.emb_dim * prototype_h * prototype_w
            else:
                assert (
                    partial_len is not None
                ), "Must set partial_len if using prototype_type='partial'"
                assert (
                    partial_overlap is not None
                ), "Must set partial_overlap if using prototype_type='partial'"
                prot_dim = self.backbone.emb_dim
        else:
            raise ValueError(f"Unknown prototype_type={self.prototype_type}")

        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, prot_dim),
            requires_grad=True,
        )
        self.emb_dim = n_prototypes

        self.__last_backbone_out = None  # (B, L, D) or (B, 1, E)
        self.__last_sim_chunk_idxs = None  # (B, P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prototype_type == "global":
            x_backbone = self.backbone(x)  # (B, E)
            x = pairwise_cosine_similarity(x_backbone, self.prototypes)  # (B, P)
            idxs = torch.zeros_like(x, dtype=torch.long)  # (B, P)
        elif self.prototype_type == "partial" and self.conv_type != "PANNS":
            # explicit windowing
            assert self.partial_len is not None
            assert self.partial_overlap is not None
            # x has shape (B, L, T), we will chunk along T axis according to partial_len/overlap
            step = int(self.partial_len * (1 - self.partial_overlap))
            x = x.unfold(2, self.partial_len, step)  # (B, L, num_chunks, partial_len)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, num_chunks, L, partial_len)
            B, N, L, P = x.shape
            x = x.view(B * N, L, P)  # (B*num_chunks, L, partial_len)
            x_backbone = self.backbone(x)  # (B*num_chunks, E)
            x = pairwise_cosine_similarity(
                x_backbone, self.prototypes
            )  # (B*num_chunks, P)
            x = x.view(B, N, -1)  # (B, num_chunks, P)
            x_backbone = x_backbone.view(B, N, -1)  # (B, num_chunks, E)
            x, idxs = x.max(1)  # (B, P) - both
        elif self.prototype_type == "partial" and self.conv_type == "PANNS":
            assert isinstance(self.backbone, PANNSEncoder)
            assert self.prototype_h is not None
            assert self.prototype_w is not None
            x_feat_map = self.backbone.local_feature_map(x)  # (B, C, H, W)
            x, x_backbone = patchwise_cosine_similarity_2d(
                feature_map=x_feat_map,
                prototypes=self.prototypes,
                prototype_h=self.prototype_h,
                prototype_w=self.prototype_w,
            )
            x, idxs = x.max(dim=1)
        else:
            raise ValueError(f"Unknown prototype_type={self.prototype_type}")

        if x_backbone.ndim == 2:
            # global embeddings, simulate chunk dim
            x_backbone = x_backbone.unsqueeze(1)  # (B, 1, E)
        self.__last_backbone_out = x_backbone  # (B, num_chunks, E) - per chunk embs
        self.__last_sim_chunk_idxs = idxs  # (B, P) - which chunk?

        return x

    def get_last_embs_and_chunks(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.__last_backbone_out is None or self.__last_sim_chunk_idxs is None:
            assert (
                self.__last_sim_chunk_idxs is None and self.__last_backbone_out is None
            ), "embs and chunks are out of sync? not sure how we got here..."
            raise ValueError(f"Model has not yet processed a batch!")
        return self.__last_backbone_out, self.__last_sim_chunk_idxs
