"""Convert Isaac Sim episode.npz into a compact training index.

This converter does not assume a particular learning target. It simply verifies
that frame paths exist and writes a JSONL file with one row per saved frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", default="datasets/isaac_franka/episode.npz")
    parser.add_argument("--output", default="datasets/isaac_franka/index.jsonl")
    args = parser.parse_args()

    episode_path = Path(args.episode)
    root = episode_path.parent
    data = np.load(episode_path, allow_pickle=True)
    frames = data["frame_paths"].tolist()
    q = data["joint_positions"]
    cube = data["cube_positions"]
    ee = data["end_effector_positions"]
    target = data["target_position"].tolist()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, rel_frame in enumerate(frames):
            record = {
                "frame": str(root / rel_frame),
                "joint_positions": q[i].tolist(),
                "cube_position": cube[i].tolist(),
                "end_effector_position": ee[i].tolist(),
                "target_position": target,
            }
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(frames)} records to {out}")


if __name__ == "__main__":
    main()
