from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _float(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize an advanced Franka training result folder.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--jump-threshold", type=float, default=0.25)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    run = Path(args.run_dir)
    metrics_path = run / "isaac_franka_advanced_metrics.csv"
    rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No metrics rows found in {metrics_path}")

    def best_by(key):
        row = min(rows, key=lambda r: _float(r, key))
        return {"epoch": int(float(row["epoch"])), key: _float(row, key)}

    sample_path = run / "plots" / "advanced_sample_prediction.json"
    jump_info = None
    if sample_path.exists():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        target = sample.get("target_future_ee", [])
        jumps = []
        for i in range(len(target) - 1):
            a, b = target[i], target[i + 1]
            jumps.append(sum((float(b[j]) - float(a[j])) ** 2 for j in range(3)) ** 0.5)
        jump_info = {
            "target_future_ee_max_jump": max(jumps) if jumps else 0.0,
            "target_future_ee_jumps_over_threshold": sum(x > args.jump_threshold for x in jumps),
            "warning": "sample target contains a large discontinuity" if jumps and max(jumps) > args.jump_threshold else "none",
        }

    report = {
        "epochs_completed": len(rows),
        "best_val_terminal_ee_dist": best_by("val_terminal_ee_dist"),
        "best_val_mean_ee_dist": best_by("val_ee_dist"),
        "best_val_q_abs": best_by("val_q_abs"),
        "final": {
            "epoch": int(float(rows[-1]["epoch"])),
            "train_terminal_ee_dist": _float(rows[-1], "train_terminal_ee_dist"),
            "val_terminal_ee_dist": _float(rows[-1], "val_terminal_ee_dist"),
            "val_ee_dist": _float(rows[-1], "val_ee_dist"),
            "val_q_abs": _float(rows[-1], "val_q_abs"),
        },
        "sample_rollout_check": jump_info,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
