# Advanced Isaac Franka Training

This v3 extension addresses the main weakness visible in a short single-episode run: the model can overfit the early part of the trajectory while the validation segment later in time remains harder. The new path uses temporal context, multi-step supervision, early stopping, and tooling for many randomized Isaac Sim episodes.

## 1. Inspect a result CSV

```bat
python scripts\analyze_training_results.py results\isaac_franka_smoke_ep50\isaac_franka_metrics.csv
```

If validation error bottoms out early and then worsens while training error continues to improve, use early stopping and collect more episodes.

## 2. Collect more Isaac Sim episodes

Run this in the Isaac Sim environment:

```bat
python scripts\isaac_sim\collect_many_franka_episodes.py --headless --episodes 20 --root-dir datasets\isaac_franka_many
python scripts\isaac_sim\merge_episode_indices.py --root-dir datasets\isaac_franka_many --output datasets\isaac_franka_many\all_index.jsonl
```

The collector launches one clean Isaac Sim app per episode. It is slower but avoids stale USD/physics state across episodes.

## 3. Train the temporal model

Run this in the lightweight PyTorch training environment:

```bat
python train_isaac_franka_advanced.py --config configs\isaac_franka_advanced.yaml --index datasets\isaac_franka_many\all_index.jsonl --epochs 80 --output-dir results\isaac_franka_advanced_many
```

For quick testing with one existing episode:

```bat
python train_isaac_franka_advanced.py --config configs\isaac_franka_advanced.yaml --index datasets\isaac_franka_v2\index.jsonl --epochs 10 --output-dir results\isaac_franka_advanced_single
```

## 4. Evaluate

```bat
python evaluate_isaac_franka_advanced.py --checkpoint results\isaac_franka_advanced_many\checkpoints\best.pt --split val --output results\isaac_franka_advanced_many\eval.json
```

## What changed from v2

- `IsaacFrankaSequenceDataset` returns K context frames/states and H future targets.
- `TemporalVisionFrankaPolicy` uses a CNN encoder plus Transformer temporal fusion.
- `sequence_prediction_loss` combines joint-delta loss, multi-step joint rollout loss, end-effector loss, terminal end-effector loss, smoothness, and joint-limit regularization.
- Optional `FrankaKinematicPrior` adds a differentiable FK consistency loss. Keep it off until you verify/calibrate the DH frame for your Isaac scene.
- Early stopping prevents the late-epoch overfitting seen on very small datasets.

## Practical settings

For one episode, keep the model modest:

```yaml
context_len: 4
horizon: 4
embed_dim: 192
num_layers: 1
```

For 20+ episodes, the default advanced config is reasonable:

```yaml
context_len: 4
horizon: 8
embed_dim: 256
num_layers: 2
```

For stronger results, collect diverse episodes instead of only training longer. A temporal holdout from a single trajectory is a distribution-shift test, not an IID validation split.
