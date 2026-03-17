from ..defines import CONV_T, RESNET_T
from ._base_classifier import BaseClassifier
from .encoders import ResNet1D, ResNet2D


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
    ):
        if conv_type == "1D":
            resnet_cls = ResNet1D(
                resnet_type=resnet_type,
                input_channels=input_channels,
            )
        elif conv_type == "2D":
            resnet_cls = ResNet2D(resnet_type=resnet_type)
        else:
            raise ValueError(f"Unknown conv_type={conv_type}")
        super().__init__(
            encoder=resnet_cls,
            n_binary_labels=n_binary_labels,
            pretrained_weights=pretrained_weights,
        )
