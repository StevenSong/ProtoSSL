from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from .htsat_core import HTSAT_Swin_Transformer
from ._base_encoder import BaseEncoder


class HTSATEncoder(BaseEncoder):
    """
    Wrapper around the original HTS-AT model.

    Input:
        x: (B, 1, T) mono waveform, already padded/cropped to 10 seconds.

    Exposes:
        - forward(x) -> (B, E) global pooled embedding
        - local_feature_map(x) -> (B, C, H, W) grouped 2D latent map
        - logmel_for_viz(x) -> (B, mel_bins, time_frames)
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
        #spec_size: int = 256,
        spec_size=(64, 1024), 
        patch_size: patch_size=(1, 4), #int = 4,
        patch_stride: tuple[int, int] = (1, 4), #tuple[int, int] = (4, 4),
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

        # Minimal config object expected by official HTS-AT file
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

        self.emb_dim = self.model.num_features
        self._last_grid_hw: tuple[int, int] | None = None

    # def _frontend(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Official HTS-AT frontend up to reshape_wav2img input.
    #     Returns shape (B, 1, time_steps, mel_bins)
    #     """
    #     if x.ndim == 3:
    #         assert x.shape[1] == 1, f"Expected mono waveform (B,1,T), got {x.shape}"
    #         x = x.squeeze(1)
    #     elif x.ndim != 2:
    #         raise ValueError(f"Expected (B,1,T) or (B,T), got {x.shape}")

    #     x = self.model.spectrogram_extractor(x)  # (B,1,time,freq)
    #     x = self.model.logmel_extractor(x)       # (B,1,time,mel)

    #     x = x.transpose(1, 3)
    #     x = self.model.bn0(x)
    #     x = x.transpose(1, 3)

    #     if self.training:
    #         x = self.model.spec_augmenter(x)
    #     return x

    # def logmel_for_viz(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Returns pre-reshape logmel for visualization.
    #     Shape: (B, mel_bins, time_frames)
    #     """
    #     x = self._frontend(x)         # (B,1,time,mel)
    #     x = x[:, 0].transpose(1, 2)   # (B, mel, time)
    #     return x.contiguous()

    # def local_feature_map(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Returns the grouped 2D latent feature map used for partial prototypes.

    #     Output:
    #         (B, C, H, W)

    #     This corresponds to the grouped latent map inside official
    #     HTS-AT's forward_features() before tscam_conv/head.
    #     """
    #     x = self._frontend(x)
    #     x = self.model.reshape_wav2img(x)  # (B,1,spec_size,spec_size_grouped)

    #     frames_num = x.shape[2]

    #     x = self.model.patch_embed(x)
    #     if self.model.ape:
    #         x = x + self.model.absolute_pos_embed
    #     x = self.model.pos_drop(x)

    #     for layer in self.model.layers:
    #         x, _attn = layer(x)

    #     x = self.model.norm(x)
    #     B, N, C = x.shape

    #     SF = frames_num // (2 ** (len(self.model.depths) - 1)) // self.model.patch_stride[0]
    #     ST = frames_num // (2 ** (len(self.model.depths) - 1)) // self.model.patch_stride[1]

    #     x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)  # (B,C,SF,ST)
    #     B, C, F, T = x.shape

    #     c_freq_bin = F // self.model.freq_ratio
    #     x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
    #     x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)

    #     self._last_grid_hw = (x.shape[2], x.shape[3])
    #     return x  # (B, C, H, W)

    # def get_last_grid_hw(self) -> tuple[int, int] | None:
    #     return self._last_grid_hw

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     feat_map = self.local_feature_map(x)   # (B,C,H,W)
    #     x = feat_map.mean(dim=(2, 3))          # (B,C)
    #     return x

##################################################################################################################################
    # def _frontend(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Official HTS-AT frontend up to reshape_wav2img input.
    #     Returns: (B, 1, time_steps, mel_bins)
    #     """
    #     if x.ndim == 3:
    #         assert x.shape[1] == 1, f"Expected mono waveform (B,1,T), got {x.shape}"
    #         x = x.squeeze(1)
    #     elif x.ndim != 2:
    #         raise ValueError(f"Expected (B,1,T) or (B,T), got {x.shape}")

    #     x = self.model.spectrogram_extractor(x)  # (B,1,time,freq)
    #     x = self.model.logmel_extractor(x)       # (B,1,time,mel)

    #     x = x.transpose(1, 3)
    #     x = self.model.bn0(x)
    #     x = x.transpose(1, 3)

    #     if self.training:
    #         x = self.model.spec_augmenter(x)

    #     return x

    # def logmel_for_viz(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Returns pre-reshape logmel for visualization.
    #     Shape: (B, mel_bins, time_frames)
    #     """
    #     x = self._frontend(x)
    #     x = x[:, 0].transpose(1, 2)   # (B, mel, time)
    #     return x.contiguous()

    # def _swin_token_features(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
    #     """
    #     Run official frontend + Swin stack up to normalized tokens.

    #     Returns:
    #         x_tokens: (B, N, C)
    #         frames_num: int
    #     """
    #     x = self._frontend(x)
    #     x = self.model.reshape_wav2img(x)

    #     frames_num = x.shape[2]

    #     x = self.model.patch_embed(x)
    #     if self.model.ape:
    #         x = x + self.model.absolute_pos_embed
    #     x = self.model.pos_drop(x)

    #     for layer in self.model.layers:
    #         x, _attn = layer(x)

    #     x = self.model.norm(x)  # (B, N, C)
    #     return x, frames_num

    # def local_feature_map(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     PRE-GROUPING 2D latent map for localized patch prototypes.

    #     Returns:
    #         (B, C, F, T)

    #     This is the map partial prototypes should use.
    #     """
    #     x, frames_num = self._swin_token_features(x)
    #     B, N, C = x.shape

    #     SF = frames_num // (2 ** (len(self.model.depths) - 1)) // self.model.patch_stride[0]
    #     ST = frames_num // (2 ** (len(self.model.depths) - 1)) // self.model.patch_stride[1]

    #     x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)  # (B,C,F,T)
    #     self._last_local_grid_hw = (x.shape[2], x.shape[3])
    #     return x

    # def get_last_grid_hw(self) -> tuple[int, int] | None:
    #     return self._last_local_grid_hw

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Generic pooled embedding from the SAME pre-grouping local map.
    #     Prototype models do not make decisions with this directly.
    #     """
    #     feat_map = self.local_feature_map(x)  # (B,C,F,T)
    #     x = feat_map.mean(dim=(2, 3))         # (B,C)
    #     return x

##################################################################################################################################

    def logmel_for_viz(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns pre-reshape logmel for visualization.
        Shape: (B, mel_bins, time_frames)
        """
        if x.ndim == 3:
            x = x.squeeze(1)

        x = self.model.spectrogram_extractor(x)
        x = self.model.logmel_extractor(x)   # (B,1,time,mel)
        x = x[:, 0].transpose(1, 2)          # (B, mel, time)
        return x.contiguous()

    def local_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        PRE-GROUPING 2D latent map for localized patch prototypes.

        Returns:
            (B, C, F, T)
        """
        if x.ndim == 3:
            assert x.shape[1] == 1, f"Expected mono waveform (B,1,T), got {x.shape}"
            x = x.squeeze(1)
        elif x.ndim != 2:
            raise ValueError(f"Expected (B,1,T) or (B,T), got {x.shape}")

        feat_map = self.model.forward_feature_map(x)   # (B, C, F, T)
        self._last_grid_hw = (feat_map.shape[2], feat_map.shape[3])
        return feat_map

    def get_last_grid_hw(self) -> tuple[int, int] | None:
        return self._last_grid_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generic pooled embedding from the SAME local map.
        Prototype models should still make decisions only from prototype activations.
        """
        feat_map = self.local_feature_map(x)   # (B, C, F, T)
        x = feat_map.mean(dim=(2, 3))          # (B, C)
        return x

##################################################################################################################################


# from __future__ import annotations

# from types import SimpleNamespace

# import torch
# import torch.nn as nn

# from .htsat_core import HTSAT_Swin_Transformer
# from ._base_encoder import BaseEncoder

# class HTSATEncoder(BaseEncoder):
#     """
#     Wrapper around modified original HTS-AT.

#     local_feature_map(x) returns the PRE-GROUPING map (B, C, F, T),
#     preserving direct correspondence to original spectrogram axes.
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
#         spec_size: tuple[int, int] = (256, 1024),
#         patch_size: int = 4,
#         patch_stride: tuple[int, int] = (4, 4),
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
#         PRE-GROUPING 2D latent map (B, C, F, T).
#         """
#         if x.ndim == 3:
#             assert x.shape[1] == 1, f"Expected mono waveform (B,1,T), got {x.shape}"
#             x = x.squeeze(1)
#         elif x.ndim != 2:
#             raise ValueError(f"Expected (B,1,T) or (B,T), got {x.shape}")

#         feat_map = self.model.forward_feature_map(x)  # (B, C, F, T)
#         self._last_grid_hw = (feat_map.shape[2], feat_map.shape[3])
#         return feat_map

#     def get_last_grid_hw(self) -> tuple[int, int] | None:
#         return self._last_grid_hw

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Generic pooled embedding from the SAME geometry-preserving local map.
#         Prototype models still make decisions only from prototype activations.
#         """
#         feat_map = self.local_feature_map(x)  # (B, C, F, T)
#         x = feat_map.mean(dim=(2, 3))         # (B, C)
#         return x