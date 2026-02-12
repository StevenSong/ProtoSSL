from ..defines import CONV_T, PROT_T, RESNET_T
from ._base_classifier import BaseClassifier
from .encoders import PrototypeEncoderWithAssignment


class PrototypeAssigner(BaseClassifier):
    @property
    def allow_extra_keys(self) -> list[str]:
        return ["proj.weight", "proj.bias", "log_temperature"]

    @property
    def allow_missing_keys(self) -> list[str]:
        return ["cls.*", "encoder.assignment_weights"]

    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes: int,
        n_prototypes_per_label: int,
        n_binary_labels: int,
        pretrained_weights: str | None = None,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
    ):
        super().__init__(
            encoder=PrototypeEncoderWithAssignment(
                resnet_type=resnet_type,
                n_prototypes=n_prototypes,
                n_prototypes_per_label=n_prototypes_per_label,
                n_labels=n_binary_labels,
                conv_type=conv_type,
                prototytpe_type=prototype_type,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
            ),
            n_binary_labels=n_binary_labels,
            pretrained_weights=pretrained_weights,
        )
