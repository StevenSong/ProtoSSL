from pass_pclr.defines import RESNET_T
from pass_pclr.models import BaseClassifier
from pass_pclr.models.encoders import PrototypeEncoder


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
