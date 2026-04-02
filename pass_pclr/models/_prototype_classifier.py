import torch
import torch.nn as nn

from ..defines import BACKBONE_T, CONV_T, PROT_T
from ._base_classifier import BaseClassifier
from .encoders import PrototypeEncoder


class PrototypeClassifier(BaseClassifier):
    @property
    def allow_extra_keys(self) -> list[str]:
        return ["proj.*", "log_temperature"]

    def __init__(
        self,
        *,  # enforce kwargs
        backbone_type: BACKBONE_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes: int,
        n_binary_labels: int,
        pretrained_weights: str | None = None,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        use_regularization_mask: bool = False,
        l1_ratio_init: float | None = None,
        alpha_init: float | None = None,
        learnable_regularization: bool = False,
        use_proto_cls_init: bool = True,  # NOTE: this overrides weights from pretrained_weights
    ):
        regularization_mask = None
        if use_regularization_mask:
            # for a given label, only regularize weights for prototypes not of that label
            # this assumes that prototypes are assigned to labels!
            # heuristic to check this is that it's divisible
            assert n_prototypes % n_binary_labels == 0
            prototypes_per_label = n_prototypes // n_binary_labels
            label_prototype_mask = torch.repeat_interleave(
                torch.eye(n_binary_labels, dtype=torch.int32),
                prototypes_per_label,
                dim=1,
            )
            regularization_mask = 1 - label_prototype_mask
        super().__init__(
            encoder=PrototypeEncoder(
                backbone_type=backbone_type,
                n_prototypes=n_prototypes,
                conv_type=conv_type,
                prototytpe_type=prototype_type,
                input_channels=input_channels,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
            ),
            n_binary_labels=n_binary_labels,
            pretrained_weights=pretrained_weights,
            regularize=True,
            regularization_mask=regularization_mask,
            l1_ratio_init=l1_ratio_init,
            alpha_init=alpha_init,
            learnable_regularization=learnable_regularization,
        )

        # apply 1 to prototype connections, -0.5 for others, 0 out bias
        if use_proto_cls_init:
            # same heuristic as above
            assert n_prototypes % n_binary_labels == 0
            ppl = n_prototypes // n_binary_labels

            # construct mask for positive connections
            mask = torch.eye(n_binary_labels, dtype=torch.int32)  # (L, L)
            mask = torch.repeat_interleave(mask, ppl, dim=1)  # (L, P)
            mask = torch.repeat_interleave(mask, 2, dim=0)  # (2L, P)

            weight = mask + (1 - mask) * -0.5
            bias = torch.zeros(n_binary_labels * 2)

            if isinstance(self.cls, nn.Linear):
                with torch.no_grad():
                    self.cls.weight.copy_(weight)
                    self.cls.bias.copy_(bias)
            else:  # assuming MultiInputLinear
                self.cls.init_weights(weight, bias)
