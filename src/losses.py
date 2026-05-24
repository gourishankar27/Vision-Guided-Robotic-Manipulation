from __future__ import annotations

import torch
from torch import Tensor


def trajectory_mse(predicted: Tensor, target: Tensor) -> Tensor:
    return torch.mean((predicted - target) ** 2)


def normalized_property_mse(predicted: Tensor, target: Tensor, mins: Tensor, maxs: Tensor) -> Tensor:
    mins = mins.to(predicted.device, predicted.dtype)
    maxs = maxs.to(predicted.device, predicted.dtype)
    pred_n = (predicted - mins) / (maxs - mins)
    targ_n = (target - mins) / (maxs - mins)
    return torch.mean((pred_n - targ_n) ** 2)


def property_mae(predicted: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.abs(predicted - target), dim=0)
