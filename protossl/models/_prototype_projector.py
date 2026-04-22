import torch
import torch.nn as nn

from ..defines import BACKBONE_T, CONV_T, PROT_T
from ._pretrained_utils import PretrainedMixin
from .encoders import PrototypeEncoder


class PrototypeProjector(PretrainedMixin, nn.Module):
    """
    Wrapper class around PrototypeEncoder with utilities for loading pretrained weights
    """

    def __init__(
        self,
        *,  # enforce kwargs
        backbone_type: BACKBONE_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes: int,
        pretrained_weights: str | None = None,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        prototype_h: int | None = None,
        prototype_w: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.encoder = PrototypeEncoder(
            backbone_type=backbone_type,
            n_prototypes=n_prototypes,
            conv_type=conv_type,
            prototype_type=prototype_type,
            input_channels=input_channels,
            partial_len=partial_len,
            partial_overlap=partial_overlap,
            prototype_h=prototype_h,
            prototype_w=prototype_w,
        )
        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

        # these might get passed in from a static config, can ignore
        kwargs.pop("label_type", None)
        kwargs.pop("use_default_weights", None)
        if len(kwargs) > 0:
            raise ValueError(f"Got unexpected keyword arguments: {list(kwargs)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def get_last_embs_and_chunks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.get_last_embs_and_chunks()

    @property
    def allow_extra_keys(self) -> list[str]:
        # fmt: off
        return [
            "proj.*", "log_temperature", # from PrototypeContraster
            "cls.weight", "cls.bias", # from PrototypeSupervisor
            "_alpha_raw", "_l1_ratio_raw", # from PrototypeClassier (cast from PrototypeAssigner)
        ]
        # fmt: on

    @property
    def allow_missing_keys(self) -> list[str]:
        return []
