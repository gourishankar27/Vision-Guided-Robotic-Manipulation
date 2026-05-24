from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from .vision_encoder import VisionEncoder


class PhysicalPropHead(nn.Module):
    """Predict physical parameters in valid ranges.

    Output order is [mass, restitution, friction]. Bounds are enforced by a
    sigmoid mapping, which avoids invalid values during early training.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        prop_mins: Iterable[float] = (0.5, 0.2, 0.02),
        prop_maxs: Iterable[float] = (3.0, 0.95, 0.6),
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        mins = torch.tensor(list(prop_mins), dtype=torch.float32)
        maxs = torch.tensor(list(prop_maxs), dtype=torch.float32)
        self.register_buffer("prop_mins", mins)
        self.register_buffer("prop_maxs", maxs)

    def forward(self, features: Tensor) -> Tensor:
        raw = self.net(features)
        normalized = torch.sigmoid(raw)
        mins = self.prop_mins.to(features.device, features.dtype)
        maxs = self.prop_maxs.to(features.device, features.dtype)
        return mins + normalized * (maxs - mins)

    def normalize(self, properties: Tensor) -> Tensor:
        mins = self.prop_mins.to(properties.device, properties.dtype)
        maxs = self.prop_maxs.to(properties.device, properties.dtype)
        return ((properties - mins) / (maxs - mins)).clamp(0.0, 1.0)


class VisionPropertyEstimator(nn.Module):
    """End-to-end image -> physical properties module."""

    def __init__(
        self,
        image_channels: int = 3,
        feature_dim: int = 128,
        base_channels: int = 32,
        hidden_dim: int = 128,
        prop_mins: Iterable[float] = (0.5, 0.2, 0.02),
        prop_maxs: Iterable[float] = (3.0, 0.95, 0.6),
    ) -> None:
        super().__init__()
        self.encoder = VisionEncoder(image_channels, feature_dim, base_channels)
        self.head = PhysicalPropHead(feature_dim, hidden_dim, prop_mins, prop_maxs)

    def forward(self, image: Tensor) -> Tensor:
        features = self.encoder(image)
        return self.head(features)

    def normalize_properties(self, properties: Tensor) -> Tensor:
        return self.head.normalize(properties)
