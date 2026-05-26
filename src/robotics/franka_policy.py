from __future__ import annotations

import torch
from torch import Tensor, nn

from src.vision_encoder import VisionEncoder


class VisionFrankaPolicy(nn.Module):
    """RGB + robot state network for Isaac Sim Franka imitation.

    The model predicts the next joint delta and the next end-effector position.
    This is a practical bridge from Isaac Sim logs into the differentiable
    physics project before replacing the kinematic head with Warp/IsaacLab
    differentiable dynamics.
    """

    def __init__(
        self,
        dof: int,
        image_channels: int = 3,
        feature_dim: int = 192,
        base_channels: int = 24,
        hidden_dim: int = 256,
        max_delta: float = 0.25,
    ) -> None:
        super().__init__()
        self.dof = int(dof)
        self.max_delta = float(max_delta)
        self.encoder = VisionEncoder(in_channels=image_channels, feature_dim=feature_dim, base_channels=base_channels)
        state_dim = self.dof + 3 + 3 + 3  # q, current EE, cube, target
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + state_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.delta_head = nn.Linear(hidden_dim, self.dof)
        self.ee_head = nn.Linear(hidden_dim, 3)

    def forward(self, frame: Tensor, q: Tensor, ee: Tensor, cube: Tensor, target: Tensor) -> dict[str, Tensor]:
        if q.ndim != 2 or q.shape[-1] != self.dof:
            raise ValueError(f"q must have shape [B, {self.dof}]")
        features = self.encoder(frame)
        state = torch.cat([q, ee, cube, target], dim=-1)
        h = self.trunk(torch.cat([features, state], dim=-1))
        delta_q = self.max_delta * torch.tanh(self.delta_head(h))
        next_ee = self.ee_head(h)
        return {"delta_q": delta_q, "next_ee": next_ee}
