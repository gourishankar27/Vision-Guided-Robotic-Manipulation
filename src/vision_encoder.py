from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.skip is None else self.skip(x)
        return self.net(x) + residual


class VisionEncoder(nn.Module):
    """Small CNN encoder.

    It plays the role of the ResNet/FPN block in the architecture diagram, but
    has no external pretrained dependency, making the project runnable offline.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 128, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(c, c, stride=1),
            ConvBlock(c, c * 2, stride=2),
            ConvBlock(c * 2, c * 4, stride=2),
            ConvBlock(c * 4, c * 4, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: Tensor) -> Tensor:
        x = self.stem(image)
        x = self.blocks(x)
        x = self.pool(x)
        return self.proj(x)
