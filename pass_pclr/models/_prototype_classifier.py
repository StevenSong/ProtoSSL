from ..defines import RESNET_T
from ._base_classifier import BaseClassifier
from .encoders import PrototypeEncoder


class PrototypeClassifier(BaseClassifier):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        n_prototypes: int,
        n_binary_labels: int,
    ):
        super().__init__(
            encoder=PrototypeEncoder(
                resnet_type=resnet_type,
                n_prototypes=n_prototypes,
            ),
            n_binary_labels=n_binary_labels,
        )
