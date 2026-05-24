# Robotics extension design

## Goal

Extend the original vision-guided differentiable physics demo from a table-top object to a robot-arm workflow:

```text
RGB observation -> vision encoder -> action/policy head -> differentiable arm/proxy physics -> target loss
```

## Components

- `train_robot_arm.py`: runnable local training loop.
- `src/robotics/planar_arm.py`: differentiable PyTorch forward kinematics and path rollout.
- `src/robotics/robot_renderer.py`: simple differentiable top-down arm renderer.
- `src/robotics/robot_dataset.py`: synthetic reach dataset for fast iteration.
- `src/engines/warp_planar_arm.py`: optional NVIDIA Warp forward-kinematics backend.
- `examples/warp_planar_arm_opt.py`: Warp Tape gradient descent demo.
- `scripts/isaac_sim/franka_reacher_collect.py`: Isaac Sim Franka RGB/trajectory collector.
- `src/splatting/gaussian_splatting.py`: minimal pure-PyTorch Gaussian splatting renderer.

## Why a proxy engine?

Isaac Sim is excellent for high-fidelity simulation and data generation, but normal deep-learning training often needs a compact differentiable module that can be unrolled thousands of times. The PyTorch proxy keeps gradients easy and fast. Isaac Sim provides visual realism and physically grounded rollouts; the differentiable proxy provides fast end-to-end gradient flow.

## Next integration steps

1. Collect Isaac Sim episodes with randomized cube poses, lighting, textures, and camera poses.
2. Add a dataset class that reads the generated `index.jsonl` and image files.
3. Train `VisionArmPolicy` first on synthetic proxy images, then fine-tune on Isaac Sim frames.
4. Replace the planar arm with a 7-DOF differentiable kinematic chain using Franka DH or URDF parameters.
5. Add a visual rendering loss using the Gaussian splat renderer or an external CUDA 3DGS rasterizer.
