from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn


class DifferentiableRenderer2D(nn.Module):
    """Tiny differentiable renderer for synthetic object frames.

    This renderer draws a soft disk on a normalized 2D table. It is used both
    for synthetic data generation and for optional image-space experiments.
    """

    def __init__(
        self,
        image_size: int = 64,
        prop_mins: Iterable[float] = (0.5, 0.2, 0.02),
        prop_maxs: Iterable[float] = (3.0, 0.95, 0.6),
        base_radius: float = 0.055,
        sharpness: float = 160.0,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.base_radius = float(base_radius)
        self.sharpness = float(sharpness)

        mins = torch.tensor(list(prop_mins), dtype=torch.float32)
        maxs = torch.tensor(list(prop_maxs), dtype=torch.float32)
        self.register_buffer("prop_mins", mins)
        self.register_buffer("prop_maxs", maxs)

        xs = torch.linspace(0.0, 1.0, self.image_size)
        ys = torch.linspace(0.0, 1.0, self.image_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid_x", grid_x[None, None])
        self.register_buffer("grid_y", grid_y[None, None])

    def normalize_properties(self, properties: Tensor) -> Tensor:
        mins = self.prop_mins.to(properties.device, properties.dtype)
        maxs = self.prop_maxs.to(properties.device, properties.dtype)
        return ((properties - mins) / (maxs - mins)).clamp(0.0, 1.0)

    def property_color(self, properties: Tensor) -> Tensor:
        """Map [mass, restitution, friction] to RGB in [0, 1]."""
        norm = self.normalize_properties(properties)
        mass, restitution, friction = norm[:, 0], norm[:, 1], norm[:, 2]
        color = torch.stack(
            [0.20 + 0.70 * mass, 0.25 + 0.65 * restitution, 0.25 + 0.65 * friction],
            dim=-1,
        )
        return color.clamp(0.0, 1.0)

    def render_frame(
        self,
        positions: Tensor,
        properties: Tensor,
        velocities: Optional[Tensor] = None,
    ) -> Tensor:
        """Render RGB frames.

        Args:
            positions: [B, 2] normalized xy object centers.
            properties: [B, 3] [mass, restitution, friction].
            velocities: currently unused hook for drawing arrows later.

        Returns:
            RGB tensor [B, 3, H, W].
        """
        if positions.ndim != 2 or positions.shape[-1] != 2:
            raise ValueError("positions must be [B, 2]")
        if properties.ndim != 2 or properties.shape[-1] != 3:
            raise ValueError("properties must be [B, 3]")

        batch_size = positions.shape[0]
        dtype = positions.dtype
        device = positions.device
        grid_x = self.grid_x.to(device=device, dtype=dtype)
        grid_y = self.grid_y.to(device=device, dtype=dtype)

        cx = positions[:, 0].view(batch_size, 1, 1, 1)
        cy = positions[:, 1].view(batch_size, 1, 1, 1)
        dist = torch.sqrt((grid_x - cx).pow(2) + (grid_y - cy).pow(2) + 1e-8)

        prop_norm = self.normalize_properties(properties)
        radius = self.base_radius * (0.75 + 0.55 * prop_norm[:, 0]).view(batch_size, 1, 1, 1)
        alpha = torch.sigmoid((radius - dist) * self.sharpness)

        # Simple dark tabletop with grid lines for spatial context.
        bg = torch.zeros(batch_size, 3, self.image_size, self.image_size, device=device, dtype=dtype)
        bg[:, 0] = 0.04
        bg[:, 1] = 0.06
        bg[:, 2] = 0.08
        grid_lines = (((grid_x * 10) % 1.0) < 0.025) | (((grid_y * 10) % 1.0) < 0.025)
        bg = bg + grid_lines.to(dtype).repeat(batch_size, 3, 1, 1) * 0.08

        color = self.property_color(properties).view(batch_size, 3, 1, 1)
        image = bg * (1.0 - alpha) + color * alpha

        # Add a small highlight for depth cue.
        highlight = torch.exp(-((grid_x - (cx - 0.018)).pow(2) + (grid_y - (cy - 0.018)).pow(2)) / 0.0009)
        image = (image + 0.20 * highlight * alpha).clamp(0.0, 1.0)
        return image

    def forward(self, positions: Tensor, properties: Tensor, velocities: Optional[Tensor] = None) -> Tensor:
        return self.render_frame(positions, properties, velocities)


def save_tensor_image(image: Tensor, path: str | Path) -> None:
    """Save a [3, H, W] or [1, 3, H, W] tensor image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 4:
        image = image[0]
    img = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_trajectory_overlay(
    observed: Tensor,
    predicted: Tensor,
    path: str | Path,
    title: str = "Trajectory overlay",
    frame: Optional[Tensor] = None,
) -> None:
    """Save a trajectory comparison plot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obs = observed.detach().cpu().numpy()
    pred = predicted.detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    if frame is not None:
        img = frame.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        plt.imshow(img, extent=(0, 1, 1, 0), alpha=0.75)
    plt.plot(obs[:, 0], obs[:, 1], linewidth=2, label="observed")
    plt.plot(pred[:, 0], pred[:, 1], "--", linewidth=2, label="predicted")
    plt.scatter(obs[0, 0], obs[0, 1], marker="o", s=40, label="start")
    plt.scatter(obs[-1, 0], obs[-1, 1], marker="x", s=60, label="observed end")
    plt.xlim(0.0, 1.0)
    plt.ylim(1.0, 0.0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
