from ..defines import RESNET_T
from ._base_classifier import BaseClassifier
from .encoders import ResNet1D


class ResNetClassifier(BaseClassifier):
    @property
    def allow_extra_keys(self) -> list[str]:
        return []

    @property
    def allow_missing_keys(self) -> list[str]:
        return []

    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        n_binary_labels: int,
        pretrained_weights: str | None = None,
    ):
        super().__init__(
            encoder=ResNet1D(resnet_type=resnet_type),
            n_binary_labels=n_binary_labels,
            pretrained_weights=pretrained_weights,
        )
