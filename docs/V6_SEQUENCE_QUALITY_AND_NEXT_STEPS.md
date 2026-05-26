# v6: Sequence quality, honest validation, and next improvements

Your bright multi-episode run reached centimeter-level validation error, but the saved demo rollout showed a target end-effector jump of roughly 0.77 m inside one 8-step future window. That is not a learnable local dynamics transition; it means windows were allowed to cross episode/reset boundaries in merged `all_index.jsonl` files.

v6 fixes this in three ways:

1. Merged JSONL files are split back into `episode_XXXX` trajectories using frame paths.
2. `split_mode: episode` holds out entire episodes for validation instead of using later windows from the same episodes.
3. Optional continuity filters reject windows with impossible step, end-effector, or joint jumps.

## Recommended workflow

Inspect sequence quality:

```bat
python scripts\analyze_isaac_sequences.py --index datasets\isaac_franka_bright_many\all_index.jsonl --require-contiguous-steps --max-ee-step 0.25 --max-q-step 1.5 --split-mode episode
```

Train with the v6 config:

```bat
python train_isaac_franka_advanced.py --config configs\isaac_franka_advanced.yaml --index datasets\isaac_franka_bright_many\all_index.jsonl --epochs 100 --output-dir results\isaac_franka_advanced_bright_many_v6
```

Evaluate and run modality ablations:

```bat
python evaluate_isaac_franka_advanced.py --checkpoint results\isaac_franka_advanced_bright_many_v6\checkpoints\best.pt --index datasets\isaac_franka_bright_many\all_index.jsonl --split val --output results\isaac_franka_advanced_bright_many_v6\eval.json

python evaluate_isaac_franka_ablation.py --checkpoint results\isaac_franka_advanced_bright_many_v6\checkpoints\best.pt --index datasets\isaac_franka_bright_many\all_index.jsonl --split val --output results\isaac_franka_advanced_bright_many_v6\vision_ablation.json
```

Summarize any completed run:

```bat
python scripts\summarize_advanced_run.py --run-dir results\isaac_franka_advanced_bright_many_v6
```

## Expected behavior

v6 validation may initially look harder than v5 because episode-level validation is stricter. That is a good thing: it measures generalization to unseen episodes rather than interpolation within the same scripted episode.

## Next feature priorities

1. Collect 50-100 brighter, more diverse Isaac episodes.
2. Enable episode-level validation as the default.
3. Add action-conditioned prediction by storing commanded joint targets/actions in the collector.
4. Add a real Franka FK prior from USD/URDF joint transforms, then enable `fk_consistency`.
5. Add closed-loop rollout evaluation with Isaac Sim replay.
6. Add multi-camera training once single-camera ablation proves vision is useful.
