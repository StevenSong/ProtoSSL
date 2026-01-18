from ..defines import RESNET_T
from ._base_classifier import BaseClassifier
from .encoders import PrototypeEncoder


class PrototypeClassifier(BaseClassifier):
    @property
    def allow_extra_keys(self) -> list[str]:
        return ["proj.weight", "proj.bias", "log_temperature"]

    @property
    def allow_missing_keys(self) -> list[str]:
        return ["cls.weight", "cls.bias"]

    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        n_prototypes: int,
        n_binary_labels: int,
        pretrained_weights: str | None = None,
    ):
        super().__init__(
            encoder=PrototypeEncoder(
                resnet_type=resnet_type,
                n_prototypes=n_prototypes,
            ),
            n_binary_labels=n_binary_labels,
            pretrained_weights=pretrained_weights,
        )
