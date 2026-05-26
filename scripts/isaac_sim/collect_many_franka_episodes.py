"""Launch the single-episode Isaac collector many times with randomized starts.

Run this in the Isaac Sim Python environment. It deliberately starts a fresh
SimulationApp for each episode by calling franka_reacher_collect.py as a child
process. This is slower than keeping one app open, but it is much more robust
across Isaac Sim versions and prevents scene cleanup/state leakage.
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect many randomized Franka episodes.")
    p.add_argument("--root-dir", default="datasets/isaac_franka_many")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--num-steps", type=int, default=650)
    p.add_argument("--save-every", type=int, default=4)
    p.add_argument("--warmup-frames", type=int, default=12)
    p.add_argument("--cube-x-range", nargs=2, type=float, default=[0.25, 0.45])
    p.add_argument("--cube-y-range", nargs=2, type=float, default=[0.15, 0.35])
    p.add_argument("--place-x-range", nargs=2, type=float, default=[-0.35, -0.15])
    p.add_argument("--place-y-range", nargs=2, type=float, default=[-0.38, -0.18])
    p.add_argument("--script", default="scripts/isaac_sim/franka_reacher_collect.py")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera-view", choices=["diag", "overhead", "front", "side"], default="diag")
    p.add_argument("--lighting", choices=["studio", "bright", "soft", "minimal"], default="studio")
    p.add_argument("--min-brightness", type=float, default=25.0)
    p.add_argument("--save-debug-preview", action="store_true")
    p.add_argument("--randomize-visuals", action="store_true")
    p.add_argument("--no-table", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    root = Path(args.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    script = Path(args.script)
    if not script.exists():
        raise FileNotFoundError(script)

    for ep in range(int(args.episodes)):
        out_dir = root / f"episode_{ep:04d}"
        cube_x = rng.uniform(*args.cube_x_range)
        cube_y = rng.uniform(*args.cube_y_range)
        place_x = rng.uniform(*args.place_x_range)
        place_y = rng.uniform(*args.place_y_range)
        cmd = [
            sys.executable,
            str(script),
            "--output-dir",
            str(out_dir),
            "--num-steps",
            str(args.num_steps),
            "--save-every",
            str(args.save_every),
            "--warmup-frames",
            str(args.warmup_frames),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--camera-view",
            str(args.camera_view),
            "--lighting",
            str(args.lighting),
            "--min-brightness",
            str(args.min_brightness),
            "--seed",
            str(args.seed + ep),
            "--cube-x",
            f"{cube_x:.5f}",
            "--cube-y",
            f"{cube_y:.5f}",
            "--place-x",
            f"{place_x:.5f}",
            "--place-y",
            f"{place_y:.5f}",
        ]
        if args.headless:
            cmd.append("--headless")
        if args.save_debug_preview:
            cmd.append("--save-debug-preview")
        if args.randomize_visuals:
            cmd.append("--randomize-visuals")
        if args.no_table:
            cmd.append("--no-table")
        print(f"[{ep + 1}/{args.episodes}] collecting {out_dir}")
        subprocess.run(cmd, check=True)
    print(f"Done. Convert with scripts/isaac_sim/merge_episode_indices.py --root-dir {root}")


if __name__ == "__main__":
    main()
