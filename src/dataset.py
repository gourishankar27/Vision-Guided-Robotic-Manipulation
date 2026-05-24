from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .physics_engine import DiffPhysicsEngine
from .renderer import DifferentiableRenderer2D


@dataclass(frozen=True)
class PropertyRanges:
    mins: Tuple[float, float, float] = (0.5, 0.2, 0.02)
    maxs: Tuple[float, float, float] = (3.0, 0.95, 0.6)


class SyntheticPhysicsDataset(Dataset):
    """Synthetic vision-guided system-identification dataset.

    Each sample contains:
      - RGB frame at t=0
      - known initial state and action sequence
      - observed trajectory generated using hidden physical properties
      - true properties, used only for metrics and optional auxiliary loss
    """

    def __init__(
        self,
        num_samples: int,
        image_size: int = 64,
        horizon: int = 120,
        dt: float = 0.02,
        seed: int = 0,
        prop_mins: Iterable[float] = (0.5, 0.2, 0.02),
        prop_maxs: Iterable[float] = (3.0, 0.95, 0.6),
        action_scale: float = 0.65,
        observation_noise_std: float = 0.001,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_samples = int(num_samples)
        self.horizon = int(horizon)
        self.image_size = int(image_size)
        self.ranges = PropertyRanges(tuple(prop_mins), tuple(prop_maxs))
        self.observation_noise_std = float(observation_noise_std)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        mins = torch.tensor(self.ranges.mins, dtype=torch.float32)
        maxs = torch.tensor(self.ranges.maxs, dtype=torch.float32)
        prop_u = torch.rand(self.num_samples, 3, generator=generator)
        true_props = mins + prop_u * (maxs - mins)

        # Avoid starting near walls so early trajectory carries clean signal.
        init_pos = 0.18 + 0.64 * torch.rand(self.num_samples, 2, generator=generator)
        init_vel = 0.10 * torch.randn(self.num_samples, 2, generator=generator)

        actions = self._make_actions(self.num_samples, self.horizon, action_scale, generator)

        engine = DiffPhysicsEngine(dt=dt, horizon=horizon)
        renderer = DifferentiableRenderer2D(
            image_size=image_size,
            prop_mins=self.ranges.mins,
            prop_maxs=self.ranges.maxs,
        )
        with torch.no_grad():
            observed_traj = engine(init_pos, init_vel, true_props, actions)
            if self.observation_noise_std > 0:
                observed_traj = observed_traj + self.observation_noise_std * torch.randn(
                    observed_traj.shape, generator=generator
                )
                observed_traj = observed_traj.clamp(0.0, 1.0)
            frame = renderer(init_pos, true_props, init_vel)

        self.samples: Dict[str, Tensor] = {
            "frame": frame.float().cpu(),
            "init_pos": init_pos.float().cpu(),
            "init_vel": init_vel.float().cpu(),
            "actions": actions.float().cpu(),
            "observed_traj": observed_traj.float().cpu(),
            "true_props": true_props.float().cpu(),
        }

    def _make_actions(
        self,
        num_samples: int,
        horizon: int,
        action_scale: float,
        generator: torch.Generator,
    ) -> Tensor:
        # A known force profile. The mass affects acceleration under this force.
        base = torch.randn(num_samples, 2, generator=generator)
        base = base / base.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        magnitude = action_scale * (0.35 + 0.65 * torch.rand(num_samples, 1, generator=generator))
        direction_force = base * magnitude

        t = torch.linspace(0, 1, horizon).view(1, horizon, 1)
        envelope = torch.exp(-2.0 * t) + 0.35 * torch.sin(3.14159 * t).pow(2)
        actions = direction_force[:, None, :] * envelope

        # Add a weak perpendicular component to make 2D trajectories richer.
        perp = torch.stack([-direction_force[:, 1], direction_force[:, 0]], dim=-1)
        actions = actions + 0.18 * perp[:, None, :] * torch.sin(2 * 3.14159 * t)
        return actions.float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {name: value[idx] for name, value in self.samples.items()}
