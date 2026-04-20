from abc import ABC, abstractmethod

import torch


class StreamingWaveformsBase(ABC):

    @abstractmethod
    def __getitem__(self, i: int) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass
