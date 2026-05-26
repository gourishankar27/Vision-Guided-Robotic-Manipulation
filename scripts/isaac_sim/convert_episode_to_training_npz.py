"""Convert Isaac Sim episode.npz into a compact training index.

This converter verifies that frame paths exist and writes a JSONL file with one
row per saved RGB/state pair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _trim_to_common_length(*arrays):
    n = min(len(x) for x in arrays)
    return n, [x[:n] for x in arrays]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", default="datasets/isaac_franka/episode.npz")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    episode_path = Path(args.episode)
    root = episode_path.parent
    output = Path(args.output) if args.output is not None else root / "index.jsonl"

    data = np.load(episode_path, allow_pickle=True)
    frames = np.asarray(data["frame_paths"]).tolist()
    q = data["joint_positions"]
    cube = data["cube_positions"]
    ee = data["end_effector_positions"]
    target = data["target_position"].tolist()
    steps = data["frame_steps"] if "frame_steps" in data.files else np.arange(len(frames))

    n, (frames, q, cube, ee, steps) = _trim_to_common_length(frames, q, cube, ee, steps)
    if n == 0:
        raise ValueError(f"No records found in {episode_path}")

    if len(data["frame_paths"]) != len(data["joint_positions"]):
        print(
            "Warning: frame/state count mismatch in episode.npz. "
            f"Using first {n} aligned-looking records. Re-collect with the updated collector for strict alignment."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    with output.open("w", encoding="utf-8") as f:
        for i, rel_frame in enumerate(frames):
            frame_path = root / str(rel_frame)
            if not frame_path.exists():
                missing += 1
            try:
                # Keep episode-level indices portable. If output is inside the
                # episode folder this becomes frames/rgb_XXXXXX.png.
                stored_frame = str(frame_path.resolve().relative_to(output.parent.resolve()))
            except ValueError:
                stored_frame = str(frame_path.resolve())
            record = {
                "frame": stored_frame,
                "step": int(steps[i]),
                "joint_positions": np.asarray(q[i], dtype=float).tolist(),
                "cube_position": np.asarray(cube[i], dtype=float).tolist(),
                "end_effector_position": np.asarray(ee[i], dtype=float).tolist(),
                "target_position": target,
            }
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {n} records to {output}")
    if missing:
        print(f"Warning: {missing} frame paths from the episode did not exist on disk.")


if __name__ == "__main__":
    main()
