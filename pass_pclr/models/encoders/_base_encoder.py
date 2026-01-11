from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    emb_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor with shape (B, ...) where B is the batch size
        :type x: torch.Tensor
        :return: embedding tensor with shape (B, H) where H is `self.emb_dim`
        :rtype: torch.Tensor
        """
        pass
