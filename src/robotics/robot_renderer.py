from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import Tensor

from .planar_arm import DifferentiablePlanarArm


def make_xy_grid(image_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.linspace(-1.0, 1.0, image_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([xx, yy], dim=-1)  # [H, W, 2]


def _segment_distance(grid: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Distance from every pixel in grid to each batched segment.

    grid: [H, W, 2]
    a, b: [B, 2]
    returns: [B, H, W]
    """
    p = grid[None, :, :, :]
    a2 = a[:, None, None, :]
    b2 = b[:, None, None, :]
    ab = b2 - a2
    denom = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    t = ((p - a2) * ab).sum(dim=-1, keepdim=True) / denom
    t = t.clamp(0.0, 1.0)
    closest = a2 + t * ab
    return torch.linalg.norm(p - closest, dim=-1)


def render_planar_arm_batch(
    joint_angles: Tensor,
    target_xy: Tensor,
    link_lengths: Iterable[float] | Tensor = (0.36, 0.28, 0.18),
    image_size: int = 96,
    arm_sigma: float = 0.018,
    joint_sigma: float = 0.025,
    target_sigma: float = 0.045,
    background: float = 0.03,
) -> Tensor:
    """Differentiably render a simple top-down arm and target image.

    The image channels intentionally encode: arm geometry, target, and joints.
    This keeps the synthetic vision task understandable while still requiring a
    CNN to parse geometry from pixels.
    """
    if target_xy.ndim != 2 or target_xy.shape[-1] != 2:
        raise ValueError("target_xy must have shape [B, 2]")
    if joint_angles.shape[0] != target_xy.shape[0]:
        raise ValueError("joint_angles and target_xy batch sizes must match")

    device = joint_angles.device
    dtype = joint_angles.dtype
    arm = DifferentiablePlanarArm(link_lengths=link_lengths).to(device=device)
    joints = arm.forward_kinematics(joint_angles).joint_positions
    batch_size = joint_angles.shape[0]
    grid = make_xy_grid(image_size, device, dtype)

    image = torch.full((batch_size, 3, image_size, image_size), float(background), device=device, dtype=dtype)

    # Arm links: cyan/blue tube-like segments.
    link_alpha = torch.zeros(batch_size, image_size, image_size, device=device, dtype=dtype)
    for idx in range(joints.shape[1] - 1):
        dist = _segment_distance(grid, joints[:, idx], joints[:, idx + 1])
        link_alpha = torch.maximum(link_alpha, torch.exp(-(dist ** 2) / (2.0 * arm_sigma ** 2)))
    image[:, 1] = torch.maximum(image[:, 1], 0.20 + 0.70 * link_alpha)
    image[:, 2] = torch.maximum(image[:, 2], 0.25 + 0.55 * link_alpha)

    # Joints/base: bright points.
    joint_alpha = torch.zeros_like(link_alpha)
    for idx in range(joints.shape[1]):
        dist = torch.linalg.norm(grid[None] - joints[:, idx, None, None, :], dim=-1)
        joint_alpha = torch.maximum(joint_alpha, torch.exp(-(dist ** 2) / (2.0 * joint_sigma ** 2)))
    image[:, 0] = torch.maximum(image[:, 0], 0.35 * joint_alpha)
    image[:, 1] = torch.maximum(image[:, 1], 0.95 * joint_alpha)
    image[:, 2] = torch.maximum(image[:, 2], 0.95 * joint_alpha)

    # Target: orange/red disk.
    target_dist = torch.linalg.norm(grid[None] - target_xy[:, None, None, :], dim=-1)
    target_alpha = torch.exp(-(target_dist ** 2) / (2.0 * target_sigma ** 2))
    image[:, 0] = torch.maximum(image[:, 0], 0.95 * target_alpha)
    image[:, 1] = torch.maximum(image[:, 1], 0.45 * target_alpha)
    image[:, 2] = torch.maximum(image[:, 2], 0.08 * target_alpha)

    return image.clamp(0.0, 1.0)
