"""Remove generated datasets, results, caches, checkpoints, and large artifacts.

Default is dry-run. Add --yes to actually delete.
Examples:
    python tools/clean_generated_data.py
    python tools/clean_generated_data.py --yes
    python tools/clean_generated_data.py --yes --keep-smoke
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIR_PATTERNS = [
    "results/*",
    "datasets/*",
    "runs",
    "outputs",
    "checkpoints",
    "wandb",
    "tensorboard",
    "lightning_logs",
    "mlruns",
    "**/__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
]
FILE_PATTERNS = [
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.onnx",
    "*.engine",
    "*.trt",
    "*.safetensors",
    "*.npz",
    "*.npy",
    "*.h5",
    "*.hdf5",
    "*.log",
    "**/rgb_*.png",
    "**/depth_*.png",
    "**/seg_*.png",
    "**/episode.npz",
    "**/index.jsonl",
    "**/all_index.jsonl",
]


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        try:
            rp = p.resolve()
        except OSError:
            rp = p
        if rp not in seen and p.exists():
            seen.add(rp)
            out.append(p)
    return sorted(out, key=lambda x: str(x).lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Actually delete files. Without this, only prints what would be deleted.")
    parser.add_argument("--keep-smoke", action="store_true", help="Keep small smoke-test result folders under results/.")
    args = parser.parse_args()

    candidates: list[Path] = []
    for pattern in DIR_PATTERNS:
        candidates.extend([p for p in ROOT.glob(pattern) if p.exists()])
    for pattern in FILE_PATTERNS:
        candidates.extend([p for p in ROOT.glob(pattern) if p.exists()])

    keep_names = {".gitkeep"}
    if args.keep_smoke:
        keep_names.update({"smoke_test", "robot_arm_smoke", "original_smoke", "splat_demo"})

    targets: list[Path] = []
    for path in unique_paths(candidates):
        rel = path.relative_to(ROOT)
        if any(part in keep_names for part in rel.parts):
            continue
        if path.name in keep_names:
            continue
        targets.append(path)

    if not targets:
        print("No generated files found.")
    else:
        action = "Deleting" if args.yes else "Would delete"
        print(f"{action} {len(targets)} generated paths under {ROOT}:")
        for p in targets[:500]:
            print("  -", p.relative_to(ROOT))
        if len(targets) > 500:
            print(f"  ... and {len(targets) - 500} more")

    if not args.yes:
        print("\nDry run only. Re-run with --yes to delete.")
        return

    for p in sorted(targets, key=lambda x: len(x.parts), reverse=True):
        if not p.exists():
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)

    for folder in [ROOT / "results", ROOT / "datasets"]:
        folder.mkdir(exist_ok=True)
        (folder / ".gitkeep").touch()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
