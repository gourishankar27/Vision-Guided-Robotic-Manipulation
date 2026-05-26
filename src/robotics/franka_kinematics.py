from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _dh_transform(theta: Tensor, d: float, a: float, alpha: float) -> Tensor:
    """Standard DH transform with batched theta, returns [..., 4, 4]."""
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    row0 = torch.stack([ct, -st * ca, st * sa, a * ct], dim=-1)
    row1 = torch.stack([st, ct * ca, -ct * sa, a * st], dim=-1)
    row2 = torch.stack([zeros, torch.full_like(theta, sa), torch.full_like(theta, ca), torch.full_like(theta, d)], dim=-1)
    row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


class FrankaKinematicPrior(nn.Module):
    """Approximate differentiable Franka Panda forward kinematics.

    This module is intentionally lightweight and PyTorch-only. It gives the
    training loop a differentiable geometric prior before a Warp/IsaacLab FK or
    dynamics backend is wired in. The affine calibration is learnable because
    Isaac Sim scenes may have a base transform/camera convention that differs
    from the nominal DH frame.
    """

    def __init__(self, learn_calibration: bool = True) -> None:
        super().__init__()
        # A commonly used approximate Panda chain. We only use the first 7 arm joints.
        self.register_buffer("d", torch.tensor([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107], dtype=torch.float32))
        self.register_buffer("a", torch.tensor([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088], dtype=torch.float32))
        self.alpha = [-math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2, 0.0]
        self.learn_calibration = bool(learn_calibration)
        if learn_calibration:
            self.scale = nn.Parameter(torch.ones(3))
            self.bias = nn.Parameter(torch.zeros(3))
        else:
            self.register_buffer("scale", torch.ones(3))
            self.register_buffer("bias", torch.zeros(3))

    def forward(self, q: Tensor) -> Tensor:
        """Return approximate end-effector xyz for q [..., dof]."""
        if q.shape[-1] < 7:
            raise ValueError("Franka FK requires at least 7 joint values")
        orig_shape = q.shape[:-1]
        q7 = q[..., :7].reshape(-1, 7)
        batch = q7.shape[0]
        T = torch.eye(4, dtype=q.dtype, device=q.device).expand(batch, 4, 4).clone()
        for i in range(7):
            Ti = _dh_transform(q7[:, i], float(self.d[i].item()), float(self.a[i].item()), self.alpha[i])
            T = T @ Ti
        pos = T[:, :3, 3]
        pos = pos * self.scale.to(pos.dtype) + self.bias.to(pos.dtype)
        return pos.reshape(*orig_shape, 3)


PANDA_JOINT_LIMITS = torch.tensor(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
        [0.0, 0.04],
        [0.0, 0.04],
    ],
    dtype=torch.float32,
)
