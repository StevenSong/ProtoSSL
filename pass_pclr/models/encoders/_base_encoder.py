from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    emb_dim: int
    ret_per_label: bool

    def __init__(self, ret_3D: bool = False):
        super().__init__()
        self.ret_per_label = ret_3D

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor with shape (B, ...) where B is the batch size
        :type x: torch.Tensor
        :return: embedding tensor with shape (B, H) (or shape (B, L, H) if ret_per_label is True) where H is `self.emb_dim`
        :rtype: torch.Tensor
        """
        pass
