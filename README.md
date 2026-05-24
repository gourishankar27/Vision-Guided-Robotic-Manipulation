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
