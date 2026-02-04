import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from ...defines import RESNET_T
from ._base_encoder import BaseEncoder


class ResNet2D(BaseEncoder):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        input_channels: int = 1,
    ):
        super().__init__()
        self.input_channels = input_channels
        match resnet_type:
            case "resnet18":
                self._make_layers(BasicBlock, [2, 2, 2, 2], input_channels)
            case "resnet34":
                self._make_layers(BasicBlock, [3, 4, 6, 3], input_channels)
            case "resnet50":
                self._make_layers(Bottleneck, [3, 4, 6, 3], input_channels)
            case "resnet101":
                self._make_layers(Bottleneck, [3, 4, 23, 3], input_channels)
            case "resnet152":
                self._make_layers(Bottleneck, [3, 8, 36, 3], input_channels)
            case _:
                raise ValueError(f"Uknown resnet_type: {resnet_type}")

    def _make_layers(self, block, layers, input_channels):
        self.inplanes = 64
        # Taken from ProtoECGNet - backbones.ResNet2D
        # Modify first convolution layer to accept (1, 12, time) input instead of RGB (3, H, W)
        self.conv1 = nn.Conv2d(
            input_channels,
            self.inplanes,
            kernel_size=(12, 7),
            stride=(1, 2),
            padding=(0, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.emb_dim = 512 * block.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.ndim == 3
        ), f"Input should be 3D (batch, leads, timesteps), got {x.shape}, ResNet2D will expand the shape for you to do 2D convolutions"
        x = x.unsqueeze(-3).expand(*x.shape[:-2], self.input_channels, *x.shape[-2:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
