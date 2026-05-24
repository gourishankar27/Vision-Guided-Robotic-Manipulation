from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PhysicsBounds:
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0


class DiffPhysicsEngine(nn.Module):
    """A compact differentiable 2D rigid-body simulator.

    State: position and velocity of a single object on a rectangular table.
    Learnable/estimated physical properties: mass, restitution, friction.

    The engine is intentionally small enough to run on CPU while still giving
    useful gradients through an unrolled symplectic Euler rollout.
    """

    def __init__(
        self,
        dt: float = 0.02,
        horizon: int = 120,
        bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        radius: float = 0.045,
        friction_acceleration: float = 1.5,
        linear_damping: float = 0.03,
        velocity_epsilon: float = 0.03,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.bounds = PhysicsBounds(*bounds)
        self.radius = float(radius)
        self.friction_acceleration = float(friction_acceleration)
        self.linear_damping = float(linear_damping)
        self.velocity_epsilon = float(velocity_epsilon)

    def forward(
        self,
        init_pos: Tensor,
        init_vel: Tensor,
        properties: Tensor,
        actions: Optional[Tensor] = None,
        horizon: Optional[int] = None,
        return_velocities: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return self.simulate(
            init_pos=init_pos,
            init_vel=init_vel,
            properties=properties,
            actions=actions,
            horizon=horizon,
            return_velocities=return_velocities,
        )

    def simulate(
        self,
        init_pos: Tensor,
        init_vel: Tensor,
        properties: Tensor,
        actions: Optional[Tensor] = None,
        horizon: Optional[int] = None,
        return_velocities: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Unroll the physics model.

        Args:
            init_pos: [B, 2] initial xy positions in normalized table coords.
            init_vel: [B, 2] initial xy velocities.
            properties: [B, 3] physical properties [mass, restitution, friction].
            actions: optional [B, T, 2] known external planar forces.
            horizon: rollout steps. Uses actions.shape[1] or self.horizon if None.
            return_velocities: also return [B, T, 2] velocities.

        Returns:
            positions [B, T, 2], and optionally velocities [B, T, 2].
        """
        if init_pos.ndim != 2 or init_pos.shape[-1] != 2:
            raise ValueError("init_pos must have shape [B, 2]")
        if init_vel.shape != init_pos.shape:
            raise ValueError("init_vel must have shape [B, 2]")
        if properties.ndim != 2 or properties.shape[-1] != 3:
            raise ValueError("properties must have shape [B, 3]")

        batch_size = init_pos.shape[0]
        device = init_pos.device
        dtype = init_pos.dtype

        if actions is not None:
            if actions.ndim != 3 or actions.shape[0] != batch_size or actions.shape[-1] != 2:
                raise ValueError("actions must have shape [B, T, 2]")
            rollout_steps = actions.shape[1] if horizon is None else int(horizon)
            actions = actions[:, :rollout_steps]
        else:
            rollout_steps = self.horizon if horizon is None else int(horizon)
            actions = torch.zeros(batch_size, rollout_steps, 2, device=device, dtype=dtype)

        mass = properties[:, 0:1].clamp_min(1e-3)
        restitution = properties[:, 1:2].clamp(0.0, 1.0)
        friction = properties[:, 2:3].clamp_min(0.0)

        pos = init_pos
        vel = init_vel
        positions = []
        velocities = []

        dt = self.dt
        xmin = self.bounds.xmin + self.radius
        xmax = self.bounds.xmax - self.radius
        ymin = self.bounds.ymin + self.radius
        ymax = self.bounds.ymax - self.radius

        for t in range(rollout_steps):
            force = actions[:, t, :]

            # Smooth Coulomb-like friction. tanh keeps gradients finite near v=0.
            friction_acc = friction * self.friction_acceleration * torch.tanh(
                vel / self.velocity_epsilon
            )
            acc = (force / mass) - friction_acc - self.linear_damping * vel

            # Symplectic Euler: update velocity before position.
            vel_next = vel + dt * acc
            pos_next = pos + dt * vel_next

            left = pos_next[:, 0:1] < xmin
            right = pos_next[:, 0:1] > xmax
            bottom = pos_next[:, 1:2] < ymin
            top = pos_next[:, 1:2] > ymax

            clamped_x = pos_next[:, 0:1].clamp(xmin, xmax)
            clamped_y = pos_next[:, 1:2].clamp(ymin, ymax)
            pos_next = torch.cat([clamped_x, clamped_y], dim=-1)

            reflect_x = left | right
            reflect_y = bottom | top
            vx = torch.where(reflect_x, -restitution * vel_next[:, 0:1], vel_next[:, 0:1])
            vy = torch.where(reflect_y, -restitution * vel_next[:, 1:2], vel_next[:, 1:2])
            vel_next = torch.cat([vx, vy], dim=-1)

            positions.append(pos_next)
            velocities.append(vel_next)
            pos = pos_next
            vel = vel_next

        pos_rollout = torch.stack(positions, dim=1)
        vel_rollout = torch.stack(velocities, dim=1)
        if return_velocities:
            return pos_rollout, vel_rollout
        return pos_rollout
