from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .franka_kinematics import PANDA_JOINT_LIMITS


@dataclass
class SequenceLossWeights:
    delta: float = 1.0
    q: float = 0.5
    ee: float = 10.0
    terminal_ee: float = 3.0
    smooth: float = 0.05
    joint_limit: float = 0.02
    fk_consistency: float = 0.0
    goal: float = 0.0


def integrate_deltas(q0: Tensor, delta_seq: Tensor) -> Tensor:
    """Integrate incremental joint deltas. q0 [B,D], delta_seq [B,H,D]."""
    return q0[:, None, :] + torch.cumsum(delta_seq, dim=1)


def _joint_limit_penalty(q_seq: Tensor) -> Tensor:
    limits = PANDA_JOINT_LIMITS.to(device=q_seq.device, dtype=q_seq.dtype)
    limits = limits[: q_seq.shape[-1]]
    lo = limits[:, 0].view(1, 1, -1)
    hi = limits[:, 1].view(1, 1, -1)
    below = F.relu(lo - q_seq)
    above = F.relu(q_seq - hi)
    return (below.square() + above.square()).mean()


def sequence_prediction_loss(
    pred: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    weights: SequenceLossWeights,
    fk_module: Optional[nn.Module] = None,
) -> tuple[Tensor, Dict[str, float]]:
    delta_pred = pred["delta_q_seq"]
    ee_pred = pred["ee_seq"]
    q_target = batch["future_joint_positions"]
    delta_target = batch["future_delta_q"]
    ee_target = batch["future_ee_positions"]
    q0 = batch["q0"]
    q_pred = integrate_deltas(q0, delta_pred)

    delta_mse = F.mse_loss(delta_pred, delta_target)
    q_mse = F.mse_loss(q_pred, q_target)
    ee_mse = F.mse_loss(ee_pred, ee_target)
    terminal_ee_mse = F.mse_loss(ee_pred[:, -1], ee_target[:, -1])
    if delta_pred.shape[1] > 1:
        smooth = F.mse_loss(delta_pred[:, 1:], delta_pred[:, :-1])
    else:
        smooth = delta_pred.new_tensor(0.0)
    joint_limit = _joint_limit_penalty(q_pred)

    fk_consistency = delta_pred.new_tensor(0.0)
    if fk_module is not None and weights.fk_consistency > 0:
        fk_ee = fk_module(q_pred)
        fk_consistency = F.mse_loss(ee_pred, fk_ee)

    goal = delta_pred.new_tensor(0.0)
    if weights.goal > 0:
        # In pick/place data the final EE should move toward the place target, not necessarily equal it.
        # Use a soft terminal distance only when explicitly enabled.
        goal = (ee_pred[:, -1] - batch["target_position"]).square().mean()

    loss = (
        weights.delta * delta_mse
        + weights.q * q_mse
        + weights.ee * ee_mse
        + weights.terminal_ee * terminal_ee_mse
        + weights.smooth * smooth
        + weights.joint_limit * joint_limit
        + weights.fk_consistency * fk_consistency
        + weights.goal * goal
    )

    with torch.no_grad():
        ee_dist = torch.linalg.norm(ee_pred - ee_target, dim=-1).mean()
        terminal_ee_dist = torch.linalg.norm(ee_pred[:, -1] - ee_target[:, -1], dim=-1).mean()
        delta_abs = torch.mean(torch.abs(delta_pred - delta_target))
        q_abs = torch.mean(torch.abs(q_pred - q_target))
    metrics = {
        "loss": float(loss.detach().cpu()),
        "delta_mse": float(delta_mse.detach().cpu()),
        "q_mse": float(q_mse.detach().cpu()),
        "ee_mse": float(ee_mse.detach().cpu()),
        "terminal_ee_mse": float(terminal_ee_mse.detach().cpu()),
        "ee_dist": float(ee_dist.cpu()),
        "terminal_ee_dist": float(terminal_ee_dist.cpu()),
        "delta_abs": float(delta_abs.cpu()),
        "q_abs": float(q_abs.cpu()),
        "smooth": float(smooth.detach().cpu()),
        "joint_limit": float(joint_limit.detach().cpu()),
        "fk_consistency": float(fk_consistency.detach().cpu()),
        "goal": float(goal.detach().cpu()),
    }
    return loss, metrics
