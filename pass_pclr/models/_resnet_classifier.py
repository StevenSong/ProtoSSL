from ..defines import CONV_T, RESNET_T
from ._base_classifier import BaseClassifier
from .encoders import HTSATEncoder, PANNSEncoder, ResNet1D, ResNet2D

class ResNetClassifier(BaseClassifier):
    @property
    def allow_extra_keys(self) -> list[str]:
        return []

    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        conv_type: CONV_T,
        n_binary_labels: int,
        input_channels: int = 12,
        pretrained_weights: str | None = None,
        audio_backbone_name: str | None = None,
    ):
        if conv_type == "1D":
            resnet_cls = ResNet1D(
                resnet_type=resnet_type,
                input_channels=input_channels,
            )
        elif conv_type == "2D":
            resnet_cls = ResNet2D(resnet_type=resnet_type)
        elif conv_type == "HTSAT":
            resnet_cls = HTSATEncoder(
                sample_rate=32000,
                clip_seconds=10.0,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                spec_size=256, #original size for blackbox version
                patch_size=4,
                patch_stride=(4, 4), #original size for blackbox version
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[4, 8, 16, 32],
                window_attn_size=8,
                num_classes=n_binary_labels,
                pretrained_checkpoint=None,
            )
        elif conv_type == "PANNS":
            resnet_cls = PANNSEncoder(
                audio_backbone_name=audio_backbone_name,
                sample_rate=32000,
                clip_seconds=10.0,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                num_classes=n_binary_labels,
            )
        else:
            raise ValueError(f"Unknown conv_type={conv_type}")
        super().__init__(
            encoder=resnet_cls,
            n_binary_labels=n_binary_labels,
            pretrained_weights=pretrained_weights,
        )
