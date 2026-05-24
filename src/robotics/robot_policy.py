from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from src.vision_encoder import VisionEncoder


class VisionArmPolicy(nn.Module):
    """CNN policy that maps an RGB frame + current joints to joint deltas."""

    def __init__(
        self,
        dof: int = 3,
        image_channels: int = 3,
        feature_dim: int = 128,
        base_channels: int = 16,
        hidden_dim: int = 128,
        max_delta: float = 1.75,
    ) -> None:
        super().__init__()
        self.dof = int(dof)
        self.max_delta = float(max_delta)
        self.encoder = VisionEncoder(
            in_channels=image_channels,
            feature_dim=feature_dim,
            base_channels=base_channels,
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim + self.dof, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.dof),
        )

    def forward(self, frame: Tensor, joint_angles: Tensor) -> Tensor:
        if joint_angles.ndim != 2 or joint_angles.shape[-1] != self.dof:
            raise ValueError(f"joint_angles must have shape [B, {self.dof}]")
        features = self.encoder(frame)
        raw = self.head(torch.cat([features, joint_angles], dim=-1))
        return self.max_delta * torch.tanh(raw)
