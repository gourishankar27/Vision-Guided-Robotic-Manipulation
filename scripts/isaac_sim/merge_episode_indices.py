from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert and merge many Isaac episode indices.")
    p.add_argument("--root-dir", default="datasets/isaac_franka_many")
    p.add_argument("--output", default=None)
    p.add_argument("--converter", default="scripts/isaac_sim/convert_episode_to_training_npz.py")
    p.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Store absolute frame paths in the merged index. By default paths are portable relative to the merged index.",
    )
    return p.parse_args()


def _candidates(raw_frame: str, episode_dir: Path, index_path: Path, root: Path) -> Iterable[Path]:
    """Candidate locations for a frame path from an episode index.

    Different versions of the converter have written paths in three styles:
    1. frames/rgb_000000.png                         relative to episode dir
    2. datasets/.../episode_0000/frames/rgb_000000.png relative to project dir
    3. absolute path

    The merger should accept all three and never prepend episode_dir twice.
    """
    frame = Path(raw_frame)
    yield frame
    if not frame.is_absolute():
        yield Path.cwd() / frame
        yield episode_dir / frame
        yield index_path.parent / frame
        yield episode_dir / "frames" / frame.name
        yield index_path.parent / "frames" / frame.name

        # If the raw path includes the root folder name, it is often relative to
        # the root's parent, not to the current episode directory.
        yield root.parent / frame

    # Last-resort repair: use the episode folder and the image filename.
    yield episode_dir / "frames" / frame.name


def _resolve_existing_frame(raw_frame: str, episode_dir: Path, index_path: Path, root: Path) -> Path:
    seen = set()
    for cand in _candidates(raw_frame, episode_dir=episode_dir, index_path=index_path, root=root):
        try:
            key = os.path.normcase(str(cand.resolve())) if cand.exists() else os.path.normcase(str(cand))
        except OSError:
            key = os.path.normcase(str(cand))
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(
        "Could not resolve frame path while merging.\n"
        f"  raw frame: {raw_frame}\n"
        f"  episode dir: {episode_dir}\n"
        f"  episode index: {index_path}\n"
        f"  root: {root}"
    )


def _portable_or_absolute(path: Path, output: Path, use_absolute: bool) -> str:
    resolved = path.resolve()
    if use_absolute:
        return str(resolved)
    try:
        return str(resolved.relative_to(output.parent.resolve()))
    except ValueError:
        return str(resolved)


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir)
    output = Path(args.output) if args.output else root / "all_index.jsonl"
    converter = Path(args.converter)
    if not root.exists():
        raise FileNotFoundError(root)
    episode_npzs = sorted(root.glob("episode_*/episode.npz"))
    if not episode_npzs:
        raise FileNotFoundError(f"No episode_*/episode.npz files found under {root}")
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as dst:
        for episode in episode_npzs:
            episode_dir = episode.parent
            index_path = episode_dir / "index.jsonl"
            if not index_path.exists():
                subprocess.run(
                    [sys.executable, str(converter), "--episode", str(episode), "--output", str(index_path)],
                    check=True,
                )
            with index_path.open("r", encoding="utf-8") as src:
                for line_number, line in enumerate(src, start=1):
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    frame_path = _resolve_existing_frame(str(raw["frame"]), episode_dir, index_path, root)
                    raw["frame"] = _portable_or_absolute(frame_path, output, args.absolute_paths)
                    raw["episode"] = episode_dir.name
                    raw["episode_index_path"] = str(index_path)
                    raw["episode_root"] = str(episode_dir)
                    dst.write(json.dumps(raw) + "\n")
                    count += 1
    print(f"Merged {count} records from {len(episode_npzs)} episodes into {output}")
    print("Frame paths verified and normalized.")


if __name__ == "__main__":
    main()
