from __future__ import annotations

from types import SimpleNamespace

import torch

from .htsat_core import HTSAT_Swin_Transformer
from ._base_encoder import BaseEncoder


class HTSATEncoder(BaseEncoder):
    """
    Blackbox wrapper around the original HTS-AT model.

    This is for baseline classification, not prototype explanations.
    It returns the original HTS-AT latent embedding from forward().
    """

    def __init__(
        self,
        *,
        sample_rate: int = 32000,
        clip_seconds: float = 10.0,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        spec_size: int = 256,
        patch_size: int = 4,
        patch_stride: tuple[int, int] = (4, 4),
        embed_dim: int = 96,
        depths: list[int] = [2, 2, 6, 2],
        num_heads: list[int] = [4, 8, 16, 32],
        window_attn_size: int = 8,
        num_classes: int = 527,
        pretrained_checkpoint: str | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.target_len = int(sample_rate * clip_seconds)

        self.cfg = SimpleNamespace(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            enable_tscam=True,
            htsat_attn_heatmap=False,
            loss_type="bce",
            enable_repeat_mode=False,
        )

        self.model = HTSAT_Swin_Transformer(
            spec_size=spec_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=1,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_attn_size,
            config=self.cfg,
            use_checkpoint=use_checkpoint,
        )

        if pretrained_checkpoint is not None:
            ckpt = torch.load(pretrained_checkpoint, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(
                "[HTSATEncoder] loaded checkpoint "
                f"{pretrained_checkpoint} | missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )

        # Original HTS-AT latent_output dim = num_features
        self.emb_dim = self.model.num_features

    def logmel_for_viz(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns pre-reshape logmel for visualization.
        Shape: (B, mel_bins, time_frames)
        """
        if x.ndim == 3:
            assert x.shape[1] == 1, f"Expected mono waveform (B,1,T), got {x.shape}"
            x = x.squeeze(1)
        elif x.ndim != 2:
            raise ValueError(f"Expected (B,1,T) or (B,T), got {x.shape}")

        x = self.model.spectrogram_extractor(x)
        x = self.model.logmel_extractor(x)   # (B,1,time,mel)
        x = x[:, 0].transpose(1, 2)          # (B, mel, time)
        return x.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the original HTS-AT latent embedding.
        """
        if x.ndim == 3:
            assert x.shape[1] == 1, f"Expected mono waveform (B,1,T), got {x.shape}"
            x = x.squeeze(1)
        elif x.ndim != 2:
            raise ValueError(f"Expected (B,1,T) or (B,T), got {x.shape}")

        out = self.model(x)
        if not isinstance(out, dict) or "latent_output" not in out:
            raise RuntimeError("Original HTS-AT forward() did not return latent_output")
        return out["latent_output"]
# from __future__ import annotations

# from types import SimpleNamespace

# import torch
# import torch.nn as nn

# from .htsat_core import HTSAT_Swin_Transformer
# from ._base_encoder import BaseEncoder


# class HTSATEncoder(BaseEncoder):
#     """
#     Wrapper around the original HTS-AT model.

#     Input:
#         x: (B, 1, T) mono waveform, already padded/cropped to 10 seconds.

#     Exposes:
#         - forward(x) -> (B, E) global pooled embedding
#         - local_feature_map(x) -> (B, C, H, W) grouped 2D latent map
#         - logmel_for_viz(x) -> (B, mel_bins, time_frames)
#     """

#     def __init__(
#         self,
#         *,
#         sample_rate: int = 32000,
#         clip_seconds: float = 10.0,
#         window_size: int = 1024,
#         hop_size: int = 320,
#         mel_bins: int = 64,
#         fmin: int = 50,
#         fmax: int = 14000,
#         #spec_size: int = 256,
#         spec_size=(64, 1024), 
#         patch_size: patch_size=(1, 4), #int = 4,
#         patch_stride: tuple[int, int] = (1, 4), #tuple[int, int] = (4, 4),
#         embed_dim: int = 96,
#         depths: list[int] = [2, 2, 6, 2],
#         num_heads: list[int] = [4, 8, 16, 32],
#         window_attn_size: int = 8,
#         num_classes: int = 527,
#         pretrained_checkpoint: str | None = None,
#         use_checkpoint: bool = False,
#     ):
#         super().__init__()

#         self.sample_rate = sample_rate
#         self.clip_seconds = clip_seconds
#         self.target_len = int(sample_rate * clip_seconds)

#         # Minimal config object expected by official HTS-AT file
#         self.cfg = SimpleNamespace(
#             sample_rate=sample_rate,
#             window_size=window_size,
#             hop_size=hop_size,
#             mel_bins=mel_bins,
#             fmin=fmin,
#             fmax=fmax,
#             enable_tscam=True,
#             htsat_attn_heatmap=False,
#             loss_type="bce",
#             enable_repeat_mode=False,
#         )

#         self.model = HTSAT_Swin_Transformer(
#             spec_size=spec_size,
#             patch_size=patch_size,
#             patch_stride=patch_stride,
#             in_chans=1,
#             num_classes=num_classes,
#             embed_dim=embed_dim,
#             depths=depths,
#             num_heads=num_heads,
#             window_size=window_attn_size,
#             config=self.cfg,
#             use_checkpoint=use_checkpoint,
#         )

#         if pretrained_checkpoint is not None:
#             ckpt = torch.load(pretrained_checkpoint, map_location="cpu")
#             state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
#             missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
#             print(
#                 "[HTSATEncoder] loaded checkpoint "
#                 f"{pretrained_checkpoint} | missing={len(missing)} "
#                 f"unexpected={len(unexpected)}"
#             )

#         self.emb_dim = self.model.num_features
#         self._last_grid_hw: tuple[int, int] | None = None

#     def logmel_for_viz(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Returns pre-reshape logmel for visualization.
#         Shape: (B, mel_bins, time_frames)
#         """
#         if x.ndim == 3:
#             x = x.squeeze(1)

#         x = self.model.spectrogram_extractor(x)
#         x = self.model.logmel_extractor(x)   # (B,1,time,mel)
#         x = x[:, 0].transpose(1, 2)          # (B, mel, time)
#         return x.contiguous()

#     def local_feature_map(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         PRE-GROUPING 2D latent map for localized patch prototypes.

#         Returns:
#             (B, C, F, T)
#         """
#         if x.ndim == 3:
#             assert x.shape[1] == 1, f"Expected mono waveform (B,1,T), got {x.shape}"
#             x = x.squeeze(1)
#         elif x.ndim != 2:
#             raise ValueError(f"Expected (B,1,T) or (B,T), got {x.shape}")

#         feat_map = self.model.forward_feature_map(x)   # (B, C, F, T)
#         self._last_grid_hw = (feat_map.shape[2], feat_map.shape[3])
#         return feat_map

#     def get_last_grid_hw(self) -> tuple[int, int] | None:
#         return self._last_grid_hw

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Generic pooled embedding from the SAME local map.
#         Prototype models should still make decisions only from prototype activations.
#         """
#         feat_map = self.local_feature_map(x)   # (B, C, F, T)
#         x = feat_map.mean(dim=(2, 3))          # (B, C)
#         return x