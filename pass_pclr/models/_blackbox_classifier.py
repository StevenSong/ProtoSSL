from ..defines import BACKBONE_T, CONV_T
from ._base_classifier import BaseClassifier
from .encoders import Net1D, ResNet1D, ResNet2D


class BlackboxClassifier(BaseClassifier):
    @property
    def allow_extra_keys(self) -> list[str]:
        return []

    def __init__(
        self,
        *,  # enforce kwargs
        backbone_type: BACKBONE_T,
        conv_type: CONV_T,
        n_binary_labels: int,
        input_channels: int = 12,
        pretrained_weights: str | None = None,
    ):
        if backbone_type == "net1d":
            if conv_type != "1D":
                raise ValueError("Must use conv_type=1D if backbone_type=net1D")
            backbone_cls = Net1D
        else:
            if conv_type == "1D":
                backbone_cls = ResNet1D
            elif conv_type == "2D":
                backbone_cls = ResNet2D
            else:
                raise ValueError(f"Unknown conv_type={conv_type}")
        super().__init__(
            encoder=backbone_cls(
                backbone_type=backbone_type, input_channels=input_channels
            ),
            n_binary_labels=n_binary_labels,
            pretrained_weights=pretrained_weights,
        )
