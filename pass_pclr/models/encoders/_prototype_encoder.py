import torch
import torch.nn as nn
import torch.nn.functional as F

from ...defines import CONV_T, PROT_T, RESNET_T
from ._base_encoder import BaseEncoder
from ._htsat_encoder import HTSATEncoder
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
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
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
    assert H_out > 0 and W_out > 0, (
        f"Prototype size ({prototype_h},{prototype_w}) exceeds feature map size ({H},{W})"
    )

    return sims, patch_vectors, (H_out, W_out)


class PrototypeEncoder(BaseEncoder):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes: int,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        prototype_h: int | None = None,
        prototype_w: int | None = None,
        audio_backbone_name: str | None = None,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.prototype_type = prototype_type

        if conv_type == "1D":
            self.resnet = ResNet1D(resnet_type=resnet_type, input_channels=input_channels)
        elif conv_type == "2D":
            self.resnet = ResNet2D(resnet_type=resnet_type)
        elif conv_type == "HTSAT":
            self.resnet = HTSATEncoder(
                sample_rate=32000,
                clip_seconds=10.0,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                spec_size=(64, 1024),
                patch_size=(1, 4),
                patch_stride=(1, 4),
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[4, 8, 16, 32],
                window_attn_size=8,
                num_classes=527,
                pretrained_checkpoint=None,
                use_checkpoint=False,
            )
        elif conv_type == "PANNS":
            self.resnet = PANNSEncoder(
                audio_backbone_name=audio_backbone_name,
                sample_rate=32000,
                clip_seconds=10.0,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                num_classes=527,
            )
        else:
            raise ValueError(f"Unknown conv_type={conv_type}")
        
        if self.prototype_type == "global":
            self.prototypes = nn.Parameter(
                torch.randn(n_prototypes, self.resnet.emb_dim),
                requires_grad=True,
            )
            self.prototype_h = None
            self.prototype_w = None

        elif self.prototype_type == "partial":
            if self.conv_type in {"HTSAT", "PANNS"}:
                assert prototype_h is not None and prototype_w is not None, (
                    "For conv_type='HTSAT' or 'PANNS' and prototype_type='partial', "
                    "set prototype_h and prototype_w explicitly."
                )
                self.prototype_h = prototype_h
                self.prototype_w = prototype_w

                patch_dim = self.resnet.emb_dim * prototype_h * prototype_w
                self.prototypes = nn.Parameter(
                    torch.randn(n_prototypes, patch_dim),
                    requires_grad=True,
                )

                # kept only for backward compatibility with existing signatures
                self.partial_len = partial_len
                self.partial_overlap = partial_overlap
            else:
                assert partial_len is not None, (
                    "Must set partial_len if using prototype_type='partial'"
                )
                assert partial_overlap is not None, (
                    "Must set partial_overlap if using prototype_type='partial'"
                )
                self.partial_len = partial_len
                self.partial_overlap = partial_overlap
                self.prototype_h = None
                self.prototype_w = None
                self.prototypes = nn.Parameter(
                    torch.randn(n_prototypes, self.resnet.emb_dim),
                    requires_grad=True,
                )
        else:
            raise ValueError(f"Unknown prototype_type={self.prototype_type}")

        self.emb_dim = n_prototypes
        self.sim_fn = pairwise_cosine_similarity

        self.__last_resnet_out = None     # (B, L, D) or (B, 1, E)
        self.__last_sim_chunk_idxs = None # (B, P)
        self.__last_local_sims = None     # (B, L, P) or (B,1,P)
        self.__last_local_grid_hw = None  # (H_out, W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prototype_type == "global":
            x_resnet = self.resnet(x)                     # (B, E)
            sims = self.sim_fn(x_resnet, self.prototypes)  # (B, P)
            idxs = torch.zeros_like(sims, dtype=torch.long)

            self.__last_resnet_out = x_resnet.unsqueeze(1)   # (B,1,E)
            self.__last_sim_chunk_idxs = idxs                # (B,P)
            self.__last_local_sims = sims.unsqueeze(1)       # (B,1,P) # necessary for HTSAT logic
            self.__last_local_grid_hw = (1, 1) # necessary for HTSAT logic
            return sims

        # Audio 2D patch partial prototypes
        if self.conv_type in {"HTSAT", "PANNS"}:
            feat_map = self.resnet.local_feature_map(x)  # (B, C, H, W)

            sims, patch_vectors, grid_hw = patchwise_cosine_similarity_2d(
                feature_map=feat_map,
                prototypes=self.prototypes,
                prototype_h=self.prototype_h,
                prototype_w=self.prototype_w,
            )

            pooled_sims, idxs = sims.max(dim=1)

            self.__last_resnet_out = patch_vectors
            self.__last_sim_chunk_idxs = idxs
            self.__last_local_sims = sims
            self.__last_local_grid_hw = grid_hw
            return pooled_sims

        # Original ECG partial prototype logic (remamed x to pooled_sims)
        step = int(self.partial_len * (1 - self.partial_overlap))
        x = x.unfold(2, self.partial_len, step)         # (B, L, N, partial_len)
        x = x.permute(0, 2, 1, 3).contiguous()         # (B, N, L, partial_len)
        B, N, L, P = x.shape
        x = x.view(B * N, L, P)                        # (B*N, L, partial_len)
        x_resnet = self.resnet(x)                      # (B*N, E)

        sims = self.sim_fn(x_resnet, self.prototypes)  # (B*N, P)
        sims = sims.view(B, N, -1)                     # (B, N, P)
        x_resnet = x_resnet.view(B, N, -1)             # (B, N, E)

        pooled_sims, idxs = sims.max(dim=1)            # both (B,P)

        self.__last_resnet_out = x_resnet              # (B,N,E)
        self.__last_sim_chunk_idxs = idxs              # (B,P)
        self.__last_local_sims = sims                  # (B,N,P)
        self.__last_local_grid_hw = (1, sims.shape[1])
        return pooled_sims

    def get_last_embs_and_chunks(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.__last_resnet_out is None or self.__last_sim_chunk_idxs is None:
            assert (
                self.__last_sim_chunk_idxs is None and self.__last_resnet_out is None
            ), "embs and chunks are out of sync? not sure how we got here..."
            raise ValueError(f"Model has not yet processed a batch!")
        return self.__last_resnet_out, self.__last_sim_chunk_idxs
    
    def get_last_local_sims(self) -> torch.Tensor:
        if self.__last_local_sims is None:
            raise ValueError("Model has not yet processed a batch!")
        return self.__last_local_sims

    def get_last_local_grid_hw(self):
        return self.__last_local_grid_hw

    def get_prototypes(self) -> torch.Tensor:
        return self.prototypes.detach()

    def get_prototype_patch_hw(self) -> tuple[int | None, int | None]:
        return self.prototype_h, self.prototype_w