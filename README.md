# Vision-Guided Differentiable Physics

A runnable PyTorch demo for **vision-guided system identification through a differentiable physics engine**.

The pipeline is:

```text
RGB frame at t=0
  -> VisionPropertyEstimator(CNN encoder + physical property head)
  -> predicted [mass, restitution, friction]
  -> DiffPhysicsEngine(symplectic Euler rollout)
  -> predicted trajectory
  -> trajectory MSE loss
  -> backprop through physics into vision encoder
```

The included dataset is synthetic and generated locally. The frame appearance encodes physical properties through color/radius so the end-to-end loop can be tested without Isaac Lab, Warp, or a robot dataset. Replace `SyntheticPhysicsDataset` with a real dataset loader when RGB-D frames and motion-capture/object tracks are available.

## Project structure

```text
vision-diff-physics/
├── src/
│   ├── physics_engine.py
│   ├── vision_encoder.py
│   ├── property_estimator.py
│   ├── dataset.py
│   ├── losses.py
│   ├── optimizer.py
│   ├── renderer.py
│   └── utils.py
├── configs/default.yaml
├── notebooks/results_viz.ipynb
├── results/
├── assets/
├── train.py
├── evaluate.py
└── requirements.txt
```

## Setup

```bash
cd vision-diff-physics
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --config configs/default.yaml
```

Fast smoke test:

```bash
python train.py --config configs/default.yaml --epochs 2 --output-dir results/smoke_test
```

Outputs:

```text
results/default_run/
├── checkpoints/best.pt
├── checkpoints/last.pt
├── metrics.csv
├── loss_curve.png
└── plots/
    ├── input_frame.png
    ├── trajectory_overlay.png
    └── property_bar.png
```

## Evaluate

```bash
python evaluate.py --config configs/default.yaml \
  --checkpoint results/default_run/checkpoints/best.pt
```

Evaluation writes metric JSON, prediction CSV, property scatter plots, and observed-vs-predicted trajectory overlays.

## Matching the architecture diagram

The default uses `horizon: 60` so it trains quickly on CPU. To match the diagram's `T=200`, edit:

```yaml
simulation:
  horizon: 200
```

## Pure system-identification mode

The default config includes a small auxiliary property loss (`property_aux_weight: 0.02`) so the synthetic demo converges quickly. For pure trajectory-only identification, use:

```yaml
training:
  property_aux_weight: 0.0
```

The main loss remains trajectory MSE in both cases.

## Robotics / Warp / Isaac Sim / 3D Gaussian Splatting extension

This repository now has two levels:

1. **Fast local differentiable proxy**: `train_robot_arm.py` trains a CNN policy from a rendered RGB arm/target frame to a joint action, then backpropagates through `src/robotics/planar_arm.py`.
2. **Simulator data source**: `scripts/isaac_sim/franka_reacher_collect.py` runs a simple Franka pick-and-place scene in Isaac Sim and saves RGB frames plus robot/cube/end-effector trajectories.

The intended workflow is:

```text
Isaac Sim Franka scene -> RGB + trajectory dataset
                         -> vision encoder / policy
                         -> differentiable proxy arm or Warp kernels
                         -> target / trajectory / render loss
                         -> optional 3D Gaussian visual loss
```

### Run the local robot-arm proxy

```bash
pip install -r requirements.txt
python train_robot_arm.py --config configs/robot_arm.yaml --epochs 3 --output-dir results/robot_arm_smoke
```

Outputs:

```text
results/robot_arm_smoke/
├── checkpoints/best.pt
├── robot_metrics.csv
├── robot_loss_curve.png
└── plots/
    ├── robot_input_frame.png
    └── robot_reach_overlay.png
```

### Optional NVIDIA Warp demo

```bash
pip install -r requirements-robotics.txt
python examples/warp_planar_arm_opt.py --device cuda
```

The core training script uses PyTorch autograd by default. Warp is added as an optional fast physics/robotics backend under `src/engines/warp_planar_arm.py` and as a native Warp-gradient optimization example in `examples/warp_planar_arm_opt.py`.

### Optional Isaac Sim Franka data collection

Use a separate Isaac Sim environment. Do not install Isaac Sim into the lightweight PyTorch venv unless your Python/CUDA stack matches NVIDIA's requirements.

```bash
python3.12 -m venv env_isaacsim
source env_isaacsim/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-isaac-sim.txt --extra-index-url https://pypi.nvidia.com
python scripts/isaac_sim/franka_reacher_collect.py --headless --output-dir datasets/isaac_franka
python scripts/isaac_sim/convert_episode_to_training_npz.py --episode datasets/isaac_franka/episode.npz
```

### Optional tiny 3D Gaussian splatting smoke test

```bash
python examples/tiny_splat_demo.py --output results/splat_demo/toy_splat.png
```

`src/splatting/gaussian_splatting.py` is a minimal differentiable PyTorch renderer for wiring visual losses into the project. For production-scale 3DGS, replace it with the official CUDA rasterizer or a maintained 3DGS package.
