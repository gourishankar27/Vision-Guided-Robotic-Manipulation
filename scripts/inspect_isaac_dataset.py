from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def resolve_frame(raw_frame: str, index_path: Path) -> Path:
    frame = Path(raw_frame)
    candidates = [frame]
    if not frame.is_absolute():
        candidates += [Path.cwd() / frame, index_path.parent / frame]
    ep = next((p for p in frame.parts if p.startswith("episode_")), None)
    if ep:
        candidates.append(index_path.parent / ep / "frames" / frame.name)
    for c in candidates:
        if c.exists():
            return c
    return frame


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect Isaac Franka JSONL dataset paths and image quality.")
    p.add_argument("--index", required=True)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dark-threshold", type=float, default=8.0, help="Mean pixel value below this 0-255 value is considered very dark.")
    args = p.parse_args()

    index_path = Path(args.index)
    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {index_path}")

    missing = []
    episodes = {}
    for i, r in enumerate(rows):
        pth = resolve_frame(str(r["frame"]), index_path)
        if not pth.exists():
            missing.append((i, r["frame"]))
        ep = r.get("episode", "unknown")
        episodes[ep] = episodes.get(ep, 0) + 1

    rng = random.Random(args.seed)
    sample_indices = rng.sample(range(len(rows)), k=min(args.samples, len(rows)))
    means = []
    stds = []
    for i in sample_indices:
        pth = resolve_frame(str(rows[i]["frame"]), index_path)
        if not pth.exists():
            continue
        arr = imageio.imread(pth)
        if arr.ndim == 3:
            arr = arr[..., :3]
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr)))

    print(f"Rows: {len(rows)}")
    print(f"Episodes: {len(episodes)}")
    print(f"Rows per episode: min={min(episodes.values())}, max={max(episodes.values())}, mean={np.mean(list(episodes.values())):.1f}")
    print(f"Missing frames: {len(missing)}")
    if missing[:5]:
        print("First missing examples:")
        for idx, frame in missing[:5]:
            print(f"  line {idx+1}: {frame}")
    if means:
        print(f"Sampled image mean brightness: mean={np.mean(means):.2f}, min={np.min(means):.2f}, max={np.max(means):.2f}")
        print(f"Sampled image std: mean={np.mean(stds):.2f}, min={np.min(stds):.2f}, max={np.max(stds):.2f}")
        dark = sum(m < args.dark_threshold for m in means)
        print(f"Very dark sampled frames: {dark}/{len(means)}")
        if dark > len(means) * 0.5:
            print("WARNING: Most frames are very dark. Recollect with improved camera/lighting before expecting strong vision results.")


if __name__ == "__main__":
    main()
