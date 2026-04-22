import torch
import torch.nn as nn

from ..defines import BACKBONE_T, CONV_T, LABEL_T, PROT_T
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
        n_labels: int,
        label_type: LABEL_T = "binary-multilabel",
        pretrained_weights: str | None = None,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        prototype_h: int | None = None,
        prototype_w: int | None = None,
        use_regularization_mask: bool = False,
        l1_ratio_init: float | None = None,
        alpha_init: float | None = None,
        learnable_regularization: bool = False,
        use_proto_cls_init: bool = False,  # NOTE: this overrides weights from pretrained_weights
    ):
        regularization_mask = None
        if use_regularization_mask:
            # for a given label, only regularize weights for prototypes not of that label
            # this assumes that prototypes are assigned to labels!
            # heuristic to check this is that it's divisible
            assert n_prototypes % n_labels == 0
            prototypes_per_label = n_prototypes // n_labels
            label_prototype_mask = torch.repeat_interleave(
                torch.eye(n_labels, dtype=torch.int32),
                prototypes_per_label,
                dim=1,
            )
            regularization_mask = 1 - label_prototype_mask
        super().__init__(
            encoder=PrototypeEncoder(
                backbone_type=backbone_type,
                n_prototypes=n_prototypes,
                conv_type=conv_type,
                prototype_type=prototype_type,
                input_channels=input_channels,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
                prototype_h=prototype_h,
                prototype_w=prototype_w,
            ),
            n_labels=n_labels,
            label_type=label_type,
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
            assert n_prototypes % n_labels == 0
            ppl = n_prototypes // n_labels

            # construct mask for positive connections
            mask = torch.eye(n_labels, dtype=torch.int32)  # (L, L)
            mask = torch.repeat_interleave(mask, ppl, dim=1)  # (L, P)
            if label_type == "binary-multilabel":
                mask = torch.repeat_interleave(mask, 2, dim=0)  # (2L, P)
                bias = torch.zeros(n_labels * 2)
            elif label_type == "multiclass":
                bias = torch.zeros(n_labels)
            else:
                raise ValueError(
                    f"Unknown how to handle prototype classifier initialization for label_type={label_type}"
                )

            weight = mask + (1 - mask) * -0.5

            if isinstance(self.cls, nn.Linear):
                with torch.no_grad():
                    self.cls.weight.copy_(weight)
                    self.cls.bias.copy_(bias)
            else:  # assuming MultiInputLinear
                self.cls.init_weights(weight, bias)
