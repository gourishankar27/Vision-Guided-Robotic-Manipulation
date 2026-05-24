from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .robot_renderer import render_planar_arm_batch


def _wrap_to_pi(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def analytic_ik_3link(target_xy: Tensor, link_lengths: Sequence[float], elbow_sign: Tensor) -> Tensor:
    """Analytic 3-link IK by merging links 2+3 into one effective link."""
    if target_xy.ndim != 2 or target_xy.shape[-1] != 2:
        raise ValueError("target_xy must have shape [B, 2]")
    l1 = float(link_lengths[0])
    l23 = float(link_lengths[1] + link_lengths[2])
    x = target_xy[:, 0]
    y = target_xy[:, 1]
    r = torch.linalg.norm(target_xy, dim=-1).clamp(1e-4, l1 + l23 - 1e-4)
    c2 = ((r ** 2 - l1 ** 2 - l23 ** 2) / (2.0 * l1 * l23)).clamp(-0.999, 0.999)
    s2 = elbow_sign * torch.sqrt((1.0 - c2 ** 2).clamp_min(0.0))
    q2 = torch.atan2(s2, c2)
    q1 = torch.atan2(y, x) - torch.atan2(l23 * s2, l1 + l23 * c2)
    q3 = torch.zeros_like(q1)
    return torch.stack([q1, q2, q3], dim=-1)


class SyntheticRobotReachDataset(Dataset):
    """Synthetic vision-to-action dataset for a planar Franka-like proxy arm.

    Each sample contains a top-down RGB frame with the arm and a target dot.
    Training uses the differentiable arm engine to make the predicted action
    move the end effector to the target.
    """

    def __init__(
        self,
        num_samples: int = 512,
        image_size: int = 96,
        link_lengths: Iterable[float] = (0.36, 0.28, 0.18),
        q_min: float = -1.4,
        q_max: float = 1.4,
        target_radius_min: float = 0.18,
        target_radius_max: float = 0.74,
        seed: int = 11,
        noise_std: float = 0.01,
        pre_render: bool = True,
    ) -> None:
        super().__init__()
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.link_lengths = tuple(float(x) for x in link_lengths)
        self.noise_std = float(noise_std)
        self.pre_render = bool(pre_render)

        generator = torch.Generator().manual_seed(int(seed))
        dof = len(self.link_lengths)
        self.joint_angles = torch.empty(self.num_samples, dof).uniform_(float(q_min), float(q_max), generator=generator)

        radius = torch.empty(self.num_samples).uniform_(target_radius_min, target_radius_max, generator=generator)
        theta = torch.empty(self.num_samples).uniform_(-math.pi, math.pi, generator=generator)
        self.target_xy = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)
        signs = torch.where(torch.rand(self.num_samples, generator=generator) > 0.5, 1.0, -1.0)
        self.target_joint_angles = analytic_ik_3link(self.target_xy, self.link_lengths, signs)
        self.oracle_delta_q = _wrap_to_pi(self.target_joint_angles - self.joint_angles)

        if self.pre_render:
            with torch.no_grad():
                frames = render_planar_arm_batch(
                    self.joint_angles,
                    self.target_xy,
                    link_lengths=self.link_lengths,
                    image_size=self.image_size,
                )
                if self.noise_std > 0.0:
                    frames = (frames + self.noise_std * torch.randn(frames.shape, generator=generator)).clamp(0.0, 1.0)
                self.frames = frames
        else:
            self.frames = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        q = self.joint_angles[index]
        target = self.target_xy[index]
        if self.frames is None:
            frame = render_planar_arm_batch(
                q[None],
                target[None],
                link_lengths=self.link_lengths,
                image_size=self.image_size,
            )[0]
        else:
            frame = self.frames[index]
        return {
            "frame": frame.float(),
            "joint_angles": q.float(),
            "target_xy": target.float(),
            "target_joint_angles": self.target_joint_angles[index].float(),
            "oracle_delta_q": self.oracle_delta_q[index].float(),
        }
