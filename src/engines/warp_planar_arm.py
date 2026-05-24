from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

try:  # Optional dependency: pip install warp-lang
    import warp as wp
except Exception:  # pragma: no cover - executed only when Warp is absent
    wp = None  # type: ignore


def is_warp_available() -> bool:
    return wp is not None


if wp is not None:
    wp.init()

    @wp.kernel
    def _planar_fk_kernel(
        q: wp.array2d(dtype=wp.float32),
        link_lengths: wp.array(dtype=wp.float32),
        out: wp.array2d(dtype=wp.float32),
        dof: int,
    ):
        i = wp.tid()
        theta = float(0.0)
        x = float(0.0)
        y = float(0.0)
        for j in range(dof):
            theta = theta + q[i, j]
            x = x + link_lengths[j] * wp.cos(theta)
            y = y + link_lengths[j] * wp.sin(theta)
        out[i, 0] = x
        out[i, 1] = y


@dataclass(frozen=True)
class WarpPlanarArmEngine:
    """Warp-accelerated forward kinematics helper.

    This class is intentionally used as an optional fast path for data generation
    and benchmarking. The default training scripts use the PyTorch arm engine so
    gradients flow through torch.autograd without requiring custom glue code.
    """

    link_lengths: Iterable[float] = (0.36, 0.28, 0.18)
    device: str = "cuda"

    def __post_init__(self) -> None:
        if wp is None:
            raise ImportError("NVIDIA Warp is not installed. Install with: pip install warp-lang")
        object.__setattr__(self, "_lengths", tuple(float(x) for x in self.link_lengths))
        if len(self._lengths) < 2:
            raise ValueError("link_lengths must have at least two entries")

    @property
    def dof(self) -> int:
        return len(self._lengths)

    def forward_kinematics_torch(self, joint_angles: Tensor) -> Tensor:
        """Run Warp FK and return a torch tensor [B, 2].

        Note: this method returns a tensor sharing memory with a Warp output
        array, but it is not inserted into PyTorch's autograd graph. Use it for
        fast rollouts/collection, or use the Warp-native optimization example
        for Warp Tape gradients.
        """
        if wp is None:
            raise ImportError("NVIDIA Warp is not installed. Install with: pip install warp-lang")
        if joint_angles.ndim != 2 or joint_angles.shape[-1] != self.dof:
            raise ValueError(f"joint_angles must have shape [B, {self.dof}]")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        else:
            device = self.device
        q = joint_angles.detach().contiguous().to(device=device, dtype=torch.float32)
        q_wp = wp.from_torch(q, dtype=wp.float32)
        lengths_t = torch.tensor(self._lengths, device=q.device, dtype=torch.float32)
        lengths_wp = wp.from_torch(lengths_t, dtype=wp.float32)
        out_wp = wp.empty((q.shape[0], 2), dtype=wp.float32, device=device)
        wp.launch(_planar_fk_kernel, dim=q.shape[0], inputs=[q_wp, lengths_wp, out_wp, self.dof], device=device)
        wp.synchronize_device(device)
        return wp.to_torch(out_wp)
