from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize Isaac Franka training CSV metrics.")
    p.add_argument("csv", help="Path to isaac_franka_metrics.csv or isaac_franka_advanced_metrics.csv")
    args = p.parse_args()
    path = Path(args.csv)
    df = pd.read_csv(path)
    print(f"Loaded {path} with {len(df)} epochs")
    for col in [c for c in df.columns if c.startswith("val_") and ("dist" in c or "loss" in c or "abs" in c)]:
        i = df[col].idxmin()
        print(f"{col:28s} best={df.loc[i, col]:.6f} epoch={int(df.loc[i, 'epoch'])}")
    if "train_ee_dist" in df and "val_ee_dist" in df:
        gap = df["val_ee_dist"].iloc[-1] - df["train_ee_dist"].iloc[-1]
        print(f"last validation-train EE gap: {gap:.6f} m")


if __name__ == "__main__":
    main()
