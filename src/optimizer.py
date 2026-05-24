from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .losses import normalized_property_mse, property_mae, trajectory_mse
from .physics_engine import DiffPhysicsEngine


@dataclass
class EpochStats:
    loss: float
    traj_mse: float
    prop_mse: float
    mass_mae: float
    restitution_mae: float
    friction_mae: float

    def as_dict(self, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_loss": self.loss,
            f"{prefix}_traj_mse": self.traj_mse,
            f"{prefix}_prop_mse": self.prop_mse,
            f"{prefix}_mass_mae": self.mass_mae,
            f"{prefix}_restitution_mae": self.restitution_mae,
            f"{prefix}_friction_mae": self.friction_mae,
        }


def move_batch(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    min_lr: float = 1e-5,
) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)


def run_epoch(
    model: nn.Module,
    engine: DiffPhysicsEngine,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    property_aux_weight: float = 0.0,
    grad_clip_norm: Optional[float] = None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)
    engine.eval()

    total_loss = 0.0
    total_traj = 0.0
    total_prop_mse = 0.0
    total_mae = torch.zeros(3, device=device)
    total_count = 0

    prop_mins = model.head.prop_mins if hasattr(model, "head") else torch.tensor([0.5, 0.2, 0.02])
    prop_maxs = model.head.prop_maxs if hasattr(model, "head") else torch.tensor([3.0, 0.95, 0.6])

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch(batch, device)
            pred_props = model(batch["frame"])
            pred_traj = engine(
                batch["init_pos"],
                batch["init_vel"],
                pred_props,
                batch["actions"],
            )
            traj_loss = trajectory_mse(pred_traj, batch["observed_traj"])
            prop_loss = normalized_property_mse(
                pred_props,
                batch["true_props"],
                prop_mins,
                prop_maxs,
            )
            loss = traj_loss + float(property_aux_weight) * prop_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            batch_size = batch["frame"].shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_traj += float(traj_loss.detach()) * batch_size
            total_prop_mse += float(prop_loss.detach()) * batch_size
            total_mae += property_mae(pred_props.detach(), batch["true_props"]).detach() * batch_size
            total_count += batch_size

    denom = max(1, total_count)
    mae = (total_mae / denom).cpu().tolist()
    return EpochStats(
        loss=total_loss / denom,
        traj_mse=total_traj / denom,
        prop_mse=total_prop_mse / denom,
        mass_mae=mae[0],
        restitution_mae=mae[1],
        friction_mae=mae[2],
    )
