# Bright Isaac Franka Collection Workflow

Your v4 run reached roughly 3 cm terminal end-effector error, but the dataset inspector reported all sampled RGB frames as very dark. That means the temporal/state model is learning, but the vision branch is not getting useful visual evidence.

v5 fixes the collection side first:

- explicit Dome, Distant, and Rect USD lights
- light-gray table/platform for visual contrast
- camera look-at pose instead of brittle Euler angles
- preview brightness report before control starts
- optional overhead rescue camera when preview is too dark
- optional visual randomization for cube/table/light exposure
- optional percentile image preprocessing and image augmentation in training
- ablation evaluation to test whether the trained model uses vision

## 1. Preview before collecting many episodes

Run inside the Isaac Sim environment:

```bat
python scripts\isaac_sim\preview_franka_camera.py --headless --output-dir datasets\isaac_franka_preview --camera-view diag --lighting studio
```

Open:

```text
datasets\isaac_franka_preview\debug_preview.png
datasets\isaac_franka_preview\frames\rgb_000000.png
```

Then inspect:

```bat
python scripts\inspect_isaac_dataset.py --index datasets\isaac_franka_preview\index.jsonl --samples 16
```

If the preview is still dark, try:

```bat
python scripts\isaac_sim\preview_franka_camera.py --headless --output-dir datasets\isaac_franka_preview_overhead --camera-view overhead --lighting bright
```

## 2. Collect bright multi-episode data

```bat
python scripts\isaac_sim\collect_many_franka_episodes.py --headless --episodes 30 --root-dir datasets\isaac_franka_bright_many --camera-view diag --lighting studio --randomize-visuals --save-debug-preview
```

Merge:

```bat
python scripts\isaac_sim\merge_episode_indices.py --root-dir datasets\isaac_franka_bright_many --output datasets\isaac_franka_bright_many\all_index.jsonl
```

Inspect:

```bat
python scripts\inspect_isaac_dataset.py --index datasets\isaac_franka_bright_many\all_index.jsonl --samples 128
```

A useful RGB dataset should usually have mean brightness well above 25 on the 0-255 scale. The exact number is not sacred; the goal is that the robot, cube, and workspace are visible.

## 3. Train with v5 preprocessing

Switch to the lightweight PyTorch training environment, then run:

```bat
python train_isaac_franka_advanced.py --config configs\isaac_franka_advanced.yaml --index datasets\isaac_franka_bright_many\all_index.jsonl --epochs 100 --output-dir results\isaac_franka_advanced_bright_many
```

## 4. Check whether vision matters

```bat
python evaluate_isaac_franka_ablation.py --checkpoint results\isaac_franka_advanced_bright_many\checkpoints\best.pt --index datasets\isaac_franka_bright_many\all_index.jsonl --split val --output results\isaac_franka_advanced_bright_many\vision_ablation.json
```

Interpretation:

- `full` is normal inference.
- `zero_vision` tests whether performance drops when RGB is removed.
- `zero_state` tests whether RGB alone carries useful scene information.

For a genuinely vision-guided model, `zero_vision` should be worse than `full` once you train on bright, varied RGB episodes.

## 5. Temporary rescue for already-collected dark data

This does not fix a wrong camera angle, but it can make under-exposed frames usable for debugging:

```bat
python tools\boost_dark_frames.py --input datasets\isaac_franka_many\all_index.jsonl --output datasets\isaac_franka_many\all_index_boosted.jsonl
python scripts\inspect_isaac_dataset.py --index datasets\isaac_franka_many\all_index_boosted.jsonl --samples 64
```

Train on the boosted index only as a diagnostic. Recollection with correct lighting/camera is the better long-term path.
