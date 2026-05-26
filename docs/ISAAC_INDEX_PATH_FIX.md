# Isaac Franka merged-index path fix

If training fails with a path like this:

```text
...\episode_0000\datasets\isaac_franka_many\episode_0000\frames\rgb_000000.png
```

the merged `all_index.jsonl` has the dataset prefix duplicated. This came from
older merge logic that prepended each episode directory to a path that was
already relative to the project root.

## Fast repair without recollecting

```bat
python tools\fix_isaac_index_paths.py --input datasets\isaac_franka_many\all_index.jsonl
```

This writes a backup:

```text
datasets\isaac_franka_many\all_index.jsonl.bak
```

Then inspect the dataset:

```bat
python scripts\inspect_isaac_dataset.py --index datasets\isaac_franka_many\all_index.jsonl --samples 64
```

Then train:

```bat
python train_isaac_franka_advanced.py --config configs\isaac_franka_advanced.yaml --index datasets\isaac_franka_many\all_index.jsonl --epochs 80 --output-dir results\isaac_franka_advanced_many
```

## Cleaner rebuild

Delete only the merged index, not the episode folders:

```bat
del datasets\isaac_franka_many\all_index.jsonl
python scripts\isaac_sim\merge_episode_indices.py --root-dir datasets\isaac_franka_many --output datasets\isaac_franka_many\all_index.jsonl
```

The updated merger verifies every frame path and stores paths as portable paths
relative to the merged index, such as:

```text
episode_0000\frames\rgb_000000.png
```

## Dark frame warning

If `inspect_isaac_dataset.py` reports that most frames are very dark, the data
will still train on joint/state inputs, but the vision branch will learn weak
features. Recollect with better camera and lighting before expecting strong
vision-guided performance.
