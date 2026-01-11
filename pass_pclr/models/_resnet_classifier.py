import torch
import torch.nn as nn
import torch.nn.functional as F

from pass_pclr.defines import RESNET_T
from pass_pclr.models import BaseClassifier
from pass_pclr.models.encoders import ResNet1D


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
