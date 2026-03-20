from ..defines import CONV_T, PROT_T, RESNET_T
from ._base_classifier import BaseClassifier
from .encoders import PrototypeEncoder

class PrototypeClassifier(BaseClassifier):
    @property
    def allow_extra_keys(self) -> list[str]:
        return ["proj.weight", "proj.bias", "log_temperature"]

    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes: int,
        n_binary_labels: int,
        pretrained_weights: str | None = None,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        prototype_h: int | None = None,
        prototype_w: int | None = None,
        audio_backbone_name: str | None = None,
    ):
        super().__init__(
            encoder=PrototypeEncoder(
                resnet_type=resnet_type,
                n_prototypes=n_prototypes,
                conv_type=conv_type,
                prototype_type=prototype_type,
                input_channels=input_channels,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
                prototype_h=prototype_h,
                prototype_w=prototype_w,
                audio_backbone_name=audio_backbone_name,
            ),
            n_binary_labels=n_binary_labels,
            pretrained_weights=pretrained_weights,
        )
