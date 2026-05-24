from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .planar_arm import DifferentiablePlanarArm


def save_arm_reach_plot(
    initial_q: Tensor,
    predicted_delta: Tensor,
    target_xy: Tensor,
    path: str | Path,
    link_lengths: Iterable[float] = (0.36, 0.28, 0.18),
    title: str = "Robot reach prediction",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arm = DifferentiablePlanarArm(link_lengths=link_lengths)
    with torch.no_grad():
        init = arm.forward_kinematics(initial_q[None].cpu()).joint_positions[0]
        pred = arm.forward_kinematics((initial_q + predicted_delta)[None].cpu()).joint_positions[0]
        target = target_xy.detach().cpu()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(init[:, 0], init[:, 1], "o--", label="initial")
    ax.plot(pred[:, 0], pred[:, 1], "o-", label="predicted final")
    ax.scatter([target[0]], [target[1]], marker="*", s=160, label="target")
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
