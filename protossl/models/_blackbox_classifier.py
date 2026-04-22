from ..defines import BACKBONE_T, CONV_T, LABEL_T
from ._base_classifier import BaseClassifier
from .encoders import Net1D, PANNSEncoder, ResNet1D, ResNet2D


class BlackboxClassifier(BaseClassifier):
    @property
    def allow_extra_keys(self) -> list[str]:
        return []

    def __init__(
        self,
        *,  # enforce kwargs
        backbone_type: BACKBONE_T,
        conv_type: CONV_T,
        n_labels: int,
        label_type: LABEL_T = "binary-multilabel",
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
            elif conv_type == "PANNS":
                backbone_cls = PANNSEncoder
            else:
                raise ValueError(f"Unknown conv_type={conv_type}")
        super().__init__(
            encoder=backbone_cls(
                backbone_type=backbone_type, input_channels=input_channels
            ),
            n_labels=n_labels,
            label_type=label_type,
            pretrained_weights=pretrained_weights,
        )
