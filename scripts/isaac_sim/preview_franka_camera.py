"""Small wrapper to collect a short bright Isaac Sim camera preview episode.

This is useful before spending time collecting 20+ episodes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Preview the v5 Franka camera/lighting setup.")
    p.add_argument("--output-dir", default="datasets/isaac_franka_preview")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--camera-view", choices=["diag", "overhead", "front", "side"], default="diag")
    p.add_argument("--lighting", choices=["studio", "bright", "soft", "minimal"], default="studio")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()
    script = Path(__file__).with_name("franka_reacher_collect.py")
    cmd = [
        sys.executable,
        str(script),
        "--output-dir",
        args.output_dir,
        "--num-steps",
        "16",
        "--save-every",
        "1",
        "--warmup-frames",
        "32",
        "--camera-view",
        args.camera_view,
        "--lighting",
        args.lighting,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--save-debug-preview",
    ]
    if args.headless:
        cmd.append("--headless")
    subprocess.run(cmd, check=True)
    converter = Path(__file__).with_name("convert_episode_to_training_npz.py")
    episode = Path(args.output_dir) / "episode.npz"
    index = Path(args.output_dir) / "index.jsonl"
    subprocess.run([sys.executable, str(converter), "--episode", str(episode), "--output", str(index)], check=True)
    print(f"Preview saved under {args.output_dir}. Open debug_preview.png and frames/rgb_000000.png.")


if __name__ == "__main__":
    main()
