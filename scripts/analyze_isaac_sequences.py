from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import List

import numpy as np

from src.robotics.isaac_franka_dataset import _load_records
from src.robotics.isaac_franka_sequence_dataset import IsaacFrankaSequenceDataset


def _norms(records, field: str) -> List[float]:
    vals = [np.asarray(getattr(r, field), dtype=float) for r in records]
    return [float(np.linalg.norm(vals[i + 1] - vals[i])) for i in range(len(vals) - 1)]


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs), q))


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze Isaac Franka temporal sequence continuity and valid training windows.")
    p.add_argument("--index", required=True)
    p.add_argument("--context-len", type=int, default=4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--require-contiguous-steps", action="store_true")
    p.add_argument("--max-ee-step", type=float, default=0.25)
    p.add_argument("--max-q-step", type=float, default=1.50)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--split-mode", default="episode", choices=["temporal", "random", "episode"])
    p.add_argument("--output", default=None)
    args = p.parse_args()

    index_path = Path(args.index)
    records = _load_records(index_path)
    # all_index.jsonl currently loads as one record list. Break by frame path episode folder when possible.
    episodes = {}
    for r in records:
        ep = next((part for part in r.frame.parts if part.startswith("episode_")), "episode_0000")
        episodes.setdefault(ep, []).append(r)

    ee_jumps_all: List[float] = []
    q_jumps_all: List[float] = []
    step_diffs_all: List[int] = []
    episode_report = {}
    for name, recs in sorted(episodes.items()):
        ee = _norms(recs, "end_effector_position")
        q = _norms(recs, "joint_positions")
        steps = [r.step for r in recs]
        diffs = [int(steps[i + 1]) - int(steps[i]) for i in range(len(steps) - 1) if steps[i] is not None and steps[i + 1] is not None]
        ee_jumps_all.extend(ee)
        q_jumps_all.extend(q)
        step_diffs_all.extend(diffs)
        episode_report[name] = {
            "rows": len(recs),
            "ee_jump_max": max(ee) if ee else 0.0,
            "ee_jump_p99": _pct(ee, 99),
            "q_jump_max": max(q) if q else 0.0,
            "q_jump_p99": _pct(q, 99),
            "step_diffs": sorted(set(diffs))[:10],
            "ee_jumps_over_threshold": int(sum(x > args.max_ee_step for x in ee)),
            "q_jumps_over_threshold": int(sum(x > args.max_q_step for x in q)),
        }

    ds_all = IsaacFrankaSequenceDataset(
        index_paths=args.index,
        image_size=0,
        context_len=args.context_len,
        horizon=args.horizon,
        stride=args.stride,
        split="all",
        split_mode=args.split_mode,
        val_fraction=args.val_fraction,
        require_contiguous_steps=args.require_contiguous_steps,
        max_ee_step=args.max_ee_step,
        max_q_step=args.max_q_step,
    )
    ds_train = IsaacFrankaSequenceDataset(
        index_paths=args.index,
        image_size=0,
        context_len=args.context_len,
        horizon=args.horizon,
        stride=args.stride,
        split="train",
        split_mode=args.split_mode,
        val_fraction=args.val_fraction,
        require_contiguous_steps=args.require_contiguous_steps,
        max_ee_step=args.max_ee_step,
        max_q_step=args.max_q_step,
    )
    ds_val = IsaacFrankaSequenceDataset(
        index_paths=args.index,
        image_size=0,
        context_len=args.context_len,
        horizon=args.horizon,
        stride=args.stride,
        split="val",
        split_mode=args.split_mode,
        val_fraction=args.val_fraction,
        require_contiguous_steps=args.require_contiguous_steps,
        max_ee_step=args.max_ee_step,
        max_q_step=args.max_q_step,
    )

    report = {
        "rows": len(records),
        "episodes": len(episodes),
        "rows_per_episode_mean": mean(len(v) for v in episodes.values()),
        "ee_jump_mean": mean(ee_jumps_all) if ee_jumps_all else 0.0,
        "ee_jump_p95": _pct(ee_jumps_all, 95),
        "ee_jump_p99": _pct(ee_jumps_all, 99),
        "ee_jump_max": max(ee_jumps_all) if ee_jumps_all else 0.0,
        "q_jump_p95": _pct(q_jumps_all, 95),
        "q_jump_p99": _pct(q_jumps_all, 99),
        "q_jump_max": max(q_jumps_all) if q_jumps_all else 0.0,
        "step_diff_values": sorted(set(step_diffs_all))[:20],
        "valid_windows_all": len(ds_all),
        "valid_windows_train": len(ds_train),
        "valid_windows_val": len(ds_val),
        "skipped_windows": ds_all.skipped_windows,
        "episodes_detail_first5": dict(list(episode_report.items())[:5]),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
