from __future__ import annotations

import torch

from ._base_encoder import BaseEncoder
from ._panns_backbones import (
    Cnn14,
    Cnn14_no_specaug,
    Cnn14_no_dropout,
    Cnn6,
    Cnn10,
    ResNet22,
    ResNet38,
    ResNet54,
    Cnn14_emb512,
    Cnn14_emb128,
    Cnn14_emb32,
    MobileNetV1,
    MobileNetV2,
    LeeNet11,
    LeeNet24,
    DaiNet19,
    Res1dNet31,
    Res1dNet51,
    Wavegram_Cnn14,
    Wavegram_Logmel_Cnn14,
    Wavegram_Logmel128_Cnn14,
    Cnn14_16k,
    Cnn14_8k,
    Cnn14_mixup_time_domain,
    Cnn14_mel32,
    Cnn14_mel128,
    Cnn14_DecisionLevelMax,
    Cnn14_DecisionLevelAvg,
    Cnn14_DecisionLevelAtt,
)


PANNS_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "Cnn14": Cnn14,
    "Cnn14_no_specaug": Cnn14_no_specaug,
    "Cnn14_no_dropout": Cnn14_no_dropout,
    "Cnn6": Cnn6,
    "Cnn10": Cnn10,
    "ResNet22": ResNet22,
    "ResNet38": ResNet38,
    "ResNet54": ResNet54,
    "Cnn14_emb512": Cnn14_emb512,
    "Cnn14_emb128": Cnn14_emb128,
    "Cnn14_emb32": Cnn14_emb32,
    "MobileNetV1": MobileNetV1,
    "MobileNetV2": MobileNetV2,
    "LeeNet11": LeeNet11,
    "LeeNet24": LeeNet24,
    "DaiNet19": DaiNet19,
    "Res1dNet31": Res1dNet31,
    "Res1dNet51": Res1dNet51,
    "Wavegram_Cnn14": Wavegram_Cnn14,
    "Wavegram_Logmel_Cnn14": Wavegram_Logmel_Cnn14,
    "Wavegram_Logmel128_Cnn14": Wavegram_Logmel128_Cnn14,
    "Cnn14_16k": Cnn14_16k,
    "Cnn14_8k": Cnn14_8k,
    "Cnn14_mixup_time_domain": Cnn14_mixup_time_domain,
    "Cnn14_mel32": Cnn14_mel32,
    "Cnn14_mel128": Cnn14_mel128,
    "Cnn14_DecisionLevelMax": Cnn14_DecisionLevelMax,
    "Cnn14_DecisionLevelAvg": Cnn14_DecisionLevelAvg,
    "Cnn14_DecisionLevelAtt": Cnn14_DecisionLevelAtt,
}


def _infer_emb_dim(model_name: str) -> int:
    if model_name == "Cnn6":
        return 512
    if model_name == "Cnn10":
        return 512
    if model_name == "ResNet22":
        return 2048
    if model_name == "ResNet38":
        return 2048
    if model_name == "ResNet54":
        return 2048
    if model_name == "Cnn14_emb512":
        return 512
    if model_name == "Cnn14_emb128":
        return 128
    if model_name == "Cnn14_emb32":
        return 32
    if model_name == "MobileNetV1":
        return 1024
    if model_name == "MobileNetV2":
        return 1024
    if model_name == "LeeNet11":
        return 512
    if model_name == "LeeNet24":
        return 1024
    if model_name == "DaiNet19":
        return 512
    if model_name == "Res1dNet31":
        return 2048
    if model_name == "Res1dNet51":
        return 2048
    return 2048


def _infer_local_channels(model_name: str) -> int:
    if model_name == "Cnn6":
        return 512
    if model_name == "Cnn10":
        return 512
    if model_name == "ResNet22":
        return 2048
    if model_name == "ResNet38":
        return 2048
    if model_name == "ResNet54":
        return 2048
    if model_name in {"MobileNetV1"}:
        return 1024
    if model_name in {"MobileNetV2"}:
        return 1280
    if model_name in {"Wavegram_Cnn14", "Wavegram_Logmel_Cnn14", "Wavegram_Logmel128_Cnn14"}:
        return 2048
    if model_name in {
        "Cnn14",
        "Cnn14_no_specaug",
        "Cnn14_no_dropout",
        "Cnn14_emb512",
        "Cnn14_emb128",
        "Cnn14_emb32",
        "Cnn14_16k",
        "Cnn14_8k",
        "Cnn14_mixup_time_domain",
        "Cnn14_mel32",
        "Cnn14_mel128",
        "Cnn14_DecisionLevelMax",
        "Cnn14_DecisionLevelAvg",
        "Cnn14_DecisionLevelAtt",
    }:
        return 2048
    raise ValueError(f"No 2D local feature map channel definition for model {model_name}")


class PANNSEncoder(BaseEncoder):
    """
    Thin wrapper around PANNS backbones.

    The backbone itself is responsible for defining:
      - _forward_latent_map(x): final latent 2D map before global pooling
      - forward(x): pooled embedding / classifier output computed from that same map
    """

    def __init__(
        self,
        *,
        audio_backbone_name: str = "Cnn14",
        sample_rate: int = 32000,
        clip_seconds: float = 10.0,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        num_classes: int = 527,
    ):
        super().__init__(ret_3D=False)

        if audio_backbone_name not in PANNS_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown PANNs model '{audio_backbone_name}'. "
                f"Available: {sorted(PANNS_MODEL_REGISTRY.keys())}"
            )

        self.model_name = audio_backbone_name
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.target_len = int(sample_rate * clip_seconds)

        model_cls = PANNS_MODEL_REGISTRY[audio_backbone_name]
        self.model = model_cls(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=num_classes,
        )

        if not hasattr(self.model, "emb_dim"):
            raise ValueError(
                f"{audio_backbone_name} must define self.emb_dim in _panns_backbones.py"
            )
        self.emb_dim = self.model.emb_dim
        self._last_grid_hw: tuple[int, int] | None = None

    def _ensure_waveform_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            assert x.shape[1] == 1, f"Expected mono waveform (B,1,T), got {x.shape}"
            x = x.squeeze(1)
        elif x.ndim != 2:
            raise ValueError(f"Expected (B,1,T) or (B,T), got {x.shape}")
        return x

    def logmel_for_viz(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_waveform_shape(x)

        if not hasattr(self.model, "spectrogram_extractor") or not hasattr(self.model, "logmel_extractor"):
            raise ValueError(f"logmel_for_viz not supported for PANNS model {self.model_name}")

        x = self.model.spectrogram_extractor(x)
        x = self.model.logmel_extractor(x)   # (B,1,time,mel)
        x = x[:, 0].transpose(1, 2).contiguous()  # (B, mel, time)
        return x

    def local_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_waveform_shape(x)

        if not hasattr(self.model, "_forward_latent_map"):
            raise ValueError(
                f"{self.model_name} does not implement _forward_latent_map(). "
                f"This must be implemented in _panns_backbones.py."
            )

        feat_map = self.model._forward_latent_map(x)   # expected (B, C, F, T)
        self._last_grid_hw = (feat_map.shape[2], feat_map.shape[3])
        return feat_map

    def get_last_grid_hw(self) -> tuple[int, int] | None:
        return self._last_grid_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_waveform_shape(x)

        if hasattr(self.model, "forward_embedding"):
            return self.model.forward_embedding(x)

        out = self.model(x)
        if isinstance(out, dict):
            if "embedding" in out:
                return out["embedding"]
            if "clipwise_output" in out:
                return out["clipwise_output"]
        raise ValueError(f"Unexpected PANNS output from {self.model_name}")