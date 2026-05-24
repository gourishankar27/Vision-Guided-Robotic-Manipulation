"""Warp-native differentiable optimization demo.

Run after installing NVIDIA Warp:
    python examples/warp_planar_arm_opt.py --device cuda

This is deliberately separate from the PyTorch training script. It demonstrates
Warp Tape gradients on a tiny IK problem, while train_robot_arm.py keeps the
main ML training path inside torch.autograd.
"""
from __future__ import annotations

import argparse

import numpy as np

try:
    import warp as wp
except Exception as exc:  # pragma: no cover
    raise SystemExit("Install NVIDIA Warp first: pip install warp-lang") from exc

wp.init()


@wp.kernel
def fk_loss_kernel(
    q: wp.array(dtype=wp.float32),
    lengths: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    theta = float(0.0)
    x = float(0.0)
    y = float(0.0)
    for j in range(3):
        theta = theta + q[j]
        x = x + lengths[j] * wp.cos(theta)
        y = y + lengths[j] * wp.sin(theta)
    dx = x - target[0]
    dy = y - target[1]
    loss[0] = dx * dx + dy * dy


@wp.kernel
def sgd_kernel(q: wp.array(dtype=wp.float32), grad: wp.array(dtype=wp.float32), lr: float):
    i = wp.tid()
    q[i] = q[i] - lr * grad[i]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.12)
    args = parser.parse_args()

    device = args.device
    q = wp.array(np.array([0.1, 0.1, 0.1], dtype=np.float32), dtype=wp.float32, device=device, requires_grad=True)
    lengths = wp.array(np.array([0.36, 0.28, 0.18], dtype=np.float32), dtype=wp.float32, device=device)
    target = wp.array(np.array([0.35, 0.35], dtype=np.float32), dtype=wp.float32, device=device)
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    for step in range(args.steps):
        loss.zero_()
        with wp.Tape() as tape:
            wp.launch(fk_loss_kernel, dim=1, inputs=[q, lengths, target, loss], device=device)
        tape.backward(loss=loss)
        wp.launch(sgd_kernel, dim=3, inputs=[q, q.grad, args.lr], device=device)
        tape.zero()
        if step % 20 == 0 or step == args.steps - 1:
            print(f"step={step:03d} loss={loss.numpy()[0]:.6f} q={q.numpy()}")


if __name__ == "__main__":
    main()
