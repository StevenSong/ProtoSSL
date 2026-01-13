from ..defines import RESNET_T
from ._base_classifier import BaseClassifier
from .encoders import ResNet1D


class ResNetClassifier(BaseClassifier):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        n_binary_labels: int,
    ):
        super().__init__(
            encoder=ResNet1D(resnet_type=resnet_type),
            n_binary_labels=n_binary_labels,
        )
