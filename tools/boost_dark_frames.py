from __future__ import annotations

import argparse
import json
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
    raise FileNotFoundError(f"Could not resolve frame {raw_frame}")


def enhance(arr: np.ndarray, low_pct: float, high_pct: float, gamma: float) -> np.ndarray:
    x = arr.astype(np.float32)
    if x.ndim == 3 and x.shape[-1] == 4:
        x = x[..., :3]
    lo = np.percentile(x, low_pct)
    hi = np.percentile(x, high_pct)
    if hi <= lo + 1e-4:
        return np.clip(x, 0, 255).astype(np.uint8)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    if gamma > 0 and abs(gamma - 1.0) > 1e-6:
        y = np.power(y, gamma)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="Create contrast-boosted copies of dark Isaac RGB frames and a new JSONL index.")
    p.add_argument("--input", required=True, help="Input index.jsonl/all_index.jsonl")
    p.add_argument("--output", default=None, help="Output index path. Defaults to <input stem>_boosted.jsonl")
    p.add_argument("--frames-dir-name", default="frames_boosted")
    p.add_argument("--low-percentile", type=float, default=0.5)
    p.add_argument("--high-percentile", type=float, default=99.5)
    p.add_argument("--gamma", type=float, default=0.65, help="<1 brightens midtones after contrast stretch.")
    args = p.parse_args()

    index_path = Path(args.input)
    output = Path(args.output) if args.output else index_path.with_name(index_path.stem + "_boosted.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {index_path}")

    converted = 0
    means_before = []
    means_after = []
    with output.open("w", encoding="utf-8") as dst:
        for row in rows:
            src = resolve_frame(str(row["frame"]), index_path)
            # Keep episode structure if present: episode_xxxx/frames_boosted/name.png
            episode = row.get("episode")
            if episode and str(episode) != "unknown":
                out_frame = index_path.parent / str(episode) / args.frames_dir_name / src.name
                rel_for_index = Path(str(episode)) / args.frames_dir_name / src.name
            else:
                out_frame = src.parent.parent / args.frames_dir_name / src.name
                rel_for_index = out_frame.resolve()
            out_frame.parent.mkdir(parents=True, exist_ok=True)
            arr = imageio.imread(src)
            boosted = enhance(arr, args.low_percentile, args.high_percentile, args.gamma)
            imageio.imwrite(out_frame, boosted)
            means_before.append(float(np.mean(arr[..., :3] if arr.ndim == 3 else arr)))
            means_after.append(float(np.mean(boosted[..., :3] if boosted.ndim == 3 else boosted)))
            row = dict(row)
            row["frame"] = str(rel_for_index)
            dst.write(json.dumps(row) + "\n")
            converted += 1

    print(f"Wrote {converted} boosted frame records to {output}")
    print(f"Mean brightness before: {np.mean(means_before):.2f}")
    print(f"Mean brightness after:  {np.mean(means_after):.2f}")


if __name__ == "__main__":
    main()
