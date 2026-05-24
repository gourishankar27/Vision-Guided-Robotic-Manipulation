from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class GaussianScene:
    means3d: Tensor          # [N, 3]
    colors: Tensor           # [N, 3] in [0, 1]
    log_scales: Tensor       # [N] log isotropic world-space radius
    opacity_logits: Tensor   # [N]


class GaussianSplatRenderer(nn.Module):
    """Tiny differentiable 3D Gaussian splatter in pure PyTorch.

    This is an educational renderer designed for integration tests and research
    scaffolding. It is not a replacement for the official CUDA tile rasterizer.
    It projects each Gaussian to image space and alpha-composites with a simple
    weighted average. The simplicity makes it useful for wiring a differentiable
    visual loss into the physics/robotics code without compiling custom CUDA.
    """

    def __init__(self, image_size: Tuple[int, int] = (96, 96), background: float = 0.02) -> None:
        super().__init__()
        self.image_size = tuple(int(x) for x in image_size)
        self.background = float(background)

    def make_default_intrinsics(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        height, width = self.image_size
        focal = 0.85 * max(height, width)
        return torch.tensor(
            [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        means3d: Tensor,
        colors: Tensor,
        log_scales: Tensor,
        opacity_logits: Tensor,
        w2c: Optional[Tensor] = None,
        intrinsics: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Render one image from a Gaussian cloud.

        Args:
            means3d: [N, 3] world coordinates.
            colors: [N, 3] RGB values. Values are sigmoid-clamped internally.
            log_scales: [N] log world radius.
            opacity_logits: [N] opacity logits.
            w2c: optional [4, 4] world-to-camera transform. Identity if omitted.
            intrinsics: optional [3, 3] pinhole camera matrix.

        Returns:
            dict with image [3, H, W], alpha [1, H, W], depth [1, H, W].
        """
        if means3d.ndim != 2 or means3d.shape[-1] != 3:
            raise ValueError("means3d must have shape [N, 3]")
        n = means3d.shape[0]
        if colors.shape != (n, 3):
            raise ValueError("colors must have shape [N, 3]")
        if log_scales.shape != (n,) or opacity_logits.shape != (n,):
            raise ValueError("log_scales and opacity_logits must have shape [N]")

        device = means3d.device
        dtype = means3d.dtype
        height, width = self.image_size
        if w2c is None:
            w2c = torch.eye(4, device=device, dtype=dtype)
        if intrinsics is None:
            intrinsics = self.make_default_intrinsics(device, dtype)

        ones = torch.ones(n, 1, device=device, dtype=dtype)
        homo = torch.cat([means3d, ones], dim=-1)
        cam = (w2c @ homo.T).T[:, :3]
        z = cam[:, 2].clamp_min(1e-4)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        u = fx * cam[:, 0] / z + cx
        v = fy * cam[:, 1] / z + cy

        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        dx = x_grid[None] - u[:, None, None]
        dy = y_grid[None] - v[:, None, None]
        sigma_px = (torch.exp(log_scales).clamp_min(1e-4) * fx / z).clamp(0.5, max(height, width) * 0.5)
        gauss = torch.exp(-0.5 * (dx ** 2 + dy ** 2) / (sigma_px[:, None, None] ** 2))
        alpha_i = torch.sigmoid(opacity_logits)[:, None, None] * gauss

        visible = (cam[:, 2] > 1e-4).to(dtype)[:, None, None]
        alpha_i = alpha_i * visible
        weights = alpha_i.clamp_min(0.0)
        denom = weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
        rgb = torch.sigmoid(colors)[:, :, None, None]
        image = (weights[:, None] * rgb).sum(dim=0) / denom
        alpha = weights.sum(dim=0, keepdim=True).clamp(0.0, 1.0)
        image = image * alpha + self.background * (1.0 - alpha)
        depth = (weights * z[:, None, None]).sum(dim=0, keepdim=True) / denom
        return {"image": image.clamp(0.0, 1.0), "alpha": alpha, "depth": depth}


def create_toy_gaussian_scene(device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> GaussianScene:
    """Create a tiny cube-like Gaussian cloud for smoke tests."""
    coords = torch.tensor(
        [
            [-0.25, -0.25, 1.6], [0.25, -0.25, 1.6], [-0.25, 0.25, 1.6], [0.25, 0.25, 1.6],
            [-0.25, -0.25, 2.0], [0.25, -0.25, 2.0], [-0.25, 0.25, 2.0], [0.25, 0.25, 2.0],
            [0.0, 0.0, 1.8],
        ],
        device=device,
        dtype=dtype,
    )
    colors = torch.tensor(
        [
            [2.0, -1.0, -1.0], [2.0, 0.5, -1.0], [-1.0, 2.0, -1.0], [-1.0, 1.0, 2.0],
            [2.0, -1.0, 0.5], [1.5, 1.0, -1.0], [-1.0, 2.0, 1.0], [0.2, 1.0, 2.0],
            [2.0, 2.0, 2.0],
        ],
        device=device,
        dtype=dtype,
    )
    log_scales = torch.full((coords.shape[0],), -3.0, device=device, dtype=dtype)
    opacity_logits = torch.full((coords.shape[0],), 2.0, device=device, dtype=dtype)
    return GaussianScene(coords, colors, log_scales, opacity_logits)
