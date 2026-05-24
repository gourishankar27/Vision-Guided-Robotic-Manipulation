from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PlanarArmState:
    """Container returned by the planar arm engine."""

    joint_positions: Tensor  # [B, D + 1, 2], includes the base at index 0
    end_effector: Tensor     # [B, 2]


class DifferentiablePlanarArm(nn.Module):
    """Small differentiable planar manipulator used as a fast training proxy.

    Isaac Sim gives realistic rigid-body/camera data. This module is the
    differentiable proxy used inside PyTorch training loops, so gradients can
    flow from an image/target loss back to the policy or vision encoder.

    Coordinates are normalized world coordinates with the arm base at (0, 0).
    Link lengths should be chosen so the full arm fits inside [-1, 1]^2.
    """

    def __init__(
        self,
        link_lengths: Iterable[float] = (0.36, 0.28, 0.18),
        joint_limit: float = 2.85,
        max_delta: float = 1.75,
        rollout_steps: int = 20,
    ) -> None:
        super().__init__()
        lengths = torch.tensor(list(link_lengths), dtype=torch.float32)
        if lengths.ndim != 1 or lengths.numel() < 2:
            raise ValueError("link_lengths must contain at least two links")
        self.register_buffer("link_lengths", lengths)
        self.joint_limit = float(joint_limit)
        self.max_delta = float(max_delta)
        self.rollout_steps = int(rollout_steps)

    @property
    def dof(self) -> int:
        return int(self.link_lengths.numel())

    @property
    def reach(self) -> float:
        return float(self.link_lengths.sum().item())

    def clamp_joint_angles(self, joint_angles: Tensor) -> Tensor:
        """Softly clamp joint angles while preserving useful gradients."""
        return self.joint_limit * torch.tanh(joint_angles / self.joint_limit)

    def scale_action(self, raw_action: Tensor) -> Tensor:
        """Map unconstrained policy output to a bounded joint delta."""
        return self.max_delta * torch.tanh(raw_action)

    def forward_kinematics(self, joint_angles: Tensor) -> PlanarArmState:
        """Compute all joint positions.

        Args:
            joint_angles: [B, D] joint angles in radians.

        Returns:
            PlanarArmState with joint_positions [B, D + 1, 2] and end_effector [B, 2].
        """
        if joint_angles.ndim != 2 or joint_angles.shape[-1] != self.dof:
            raise ValueError(f"joint_angles must have shape [B, {self.dof}]")

        angles = self.clamp_joint_angles(joint_angles)
        cumulative = torch.cumsum(angles, dim=-1)
        dx = self.link_lengths.to(joint_angles.device, joint_angles.dtype) * torch.cos(cumulative)
        dy = self.link_lengths.to(joint_angles.device, joint_angles.dtype) * torch.sin(cumulative)
        displacements = torch.stack([dx, dy], dim=-1)
        joints_no_base = torch.cumsum(displacements, dim=1)
        base = torch.zeros(joint_angles.shape[0], 1, 2, device=joint_angles.device, dtype=joint_angles.dtype)
        joint_positions = torch.cat([base, joints_no_base], dim=1)
        return PlanarArmState(joint_positions=joint_positions, end_effector=joint_positions[:, -1])

    def rollout(
        self,
        initial_joint_angles: Tensor,
        joint_delta: Tensor,
        steps: Optional[int] = None,
    ) -> Tensor:
        """Interpolate from q0 to q0 + delta and return end-effector path.

        Args:
            initial_joint_angles: [B, D]
            joint_delta: [B, D]
            steps: number of interpolation steps.

        Returns:
            end_effector_path: [B, T, 2]
        """
        if initial_joint_angles.shape != joint_delta.shape:
            raise ValueError("initial_joint_angles and joint_delta must have matching shape")
        steps = self.rollout_steps if steps is None else int(steps)
        alphas = torch.linspace(0.0, 1.0, steps, device=initial_joint_angles.device, dtype=initial_joint_angles.dtype)
        q_path = initial_joint_angles[:, None, :] + alphas[None, :, None] * joint_delta[:, None, :]
        flat = q_path.reshape(-1, self.dof)
        ee = self.forward_kinematics(flat).end_effector
        return ee.reshape(initial_joint_angles.shape[0], steps, 2)

    def final_end_effector(self, initial_joint_angles: Tensor, joint_delta: Tensor) -> Tensor:
        q_final = initial_joint_angles + joint_delta
        return self.forward_kinematics(q_final).end_effector
