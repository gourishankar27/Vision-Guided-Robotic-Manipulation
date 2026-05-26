from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.robotics.franka_advanced_loss import SequenceLossWeights, sequence_prediction_loss
from src.robotics.franka_temporal_policy import TemporalVisionFrankaPolicy
from src.robotics.isaac_franka_sequence_dataset import IsaacFrankaSequenceDataset
from src.utils import get_device, load_config


def _build_dataset(config: Dict[str, Any], split: str) -> IsaacFrankaSequenceDataset:
    data_cfg = config["data"]
    index_paths = data_cfg.get("index_paths") or data_cfg["index_path"]
    return IsaacFrankaSequenceDataset(
        index_paths=index_paths,
        image_size=int(data_cfg.get("image_size", 128)),
        context_len=int(data_cfg.get("context_len", 4)),
        horizon=int(data_cfg.get("horizon", 8)),
        stride=int(data_cfg.get("stride", 1)),
        split=split,
        val_fraction=float(data_cfg.get("val_fraction", 0.2)),
        split_mode=str(data_cfg.get("split_mode", "temporal")),
        max_windows=data_cfg.get("max_windows"),
        cache_images=bool(data_cfg.get("cache_images", False)),
        image_preprocess=str(data_cfg.get("image_preprocess", "none")),
    )


def _build_model(config: Dict[str, Any], dof: int, device: torch.device) -> TemporalVisionFrankaPolicy:
    data_cfg = config["data"]
    model_cfg = config["model"]
    return TemporalVisionFrankaPolicy(
        dof=dof,
        context_len=int(data_cfg.get("context_len", 4)),
        horizon=int(data_cfg.get("horizon", 8)),
        feature_dim=int(model_cfg.get("feature_dim", 192)),
        base_channels=int(model_cfg.get("base_channels", 24)),
        embed_dim=int(model_cfg.get("embed_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.05)),
        max_delta=float(model_cfg.get("max_delta", 0.25)),
        vision_dropout=float(model_cfg.get("vision_dropout", 0.0)),
        state_dropout=float(model_cfg.get("state_dropout", 0.0)),
    ).to(device)


def _apply_ablation(batch: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
    if mode == "full":
        return batch
    batch = dict(batch)
    if mode == "zero_vision":
        batch["frames"] = torch.zeros_like(batch["frames"])
    elif mode == "zero_q":
        batch["q_context"] = torch.zeros_like(batch["q_context"])
    elif mode == "zero_ee_cube":
        batch["ee_context"] = torch.zeros_like(batch["ee_context"])
        batch["cube_context"] = torch.zeros_like(batch["cube_context"])
    elif mode == "zero_state":
        batch["q_context"] = torch.zeros_like(batch["q_context"])
        batch["ee_context"] = torch.zeros_like(batch["ee_context"])
        batch["cube_context"] = torch.zeros_like(batch["cube_context"])
        batch["target_position"] = torch.zeros_like(batch["target_position"])
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")
    return batch


def _eval_mode(model, loader, device, mode: str) -> Dict[str, float]:
    weights = SequenceLossWeights()
    totals: Dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch = _apply_ablation(batch, mode)
            pred = model(batch["frames"], batch["q_context"], batch["ee_context"], batch["cube_context"], batch["target_position"])
            _, metrics = sequence_prediction_loss(pred, batch, weights, fk_module=None)
            bs = int(batch["frames"].shape[0])
            n += bs
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v) * bs
    out = {k: v / max(n, 1) for k, v in totals.items()}
    out["num_windows"] = float(n)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate whether the trained Isaac Franka model uses vision or only state.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--index", default=None)
    p.add_argument("--split", default="val", choices=["all", "train", "val"])
    p.add_argument("--output", default=None)
    p.add_argument("--modes", default="full,zero_vision,zero_q,zero_ee_cube,zero_state")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config: Dict[str, Any] = load_config(args.config) if args.config else ckpt["config"]
    if args.index is not None:
        config["data"]["index_path"] = args.index
        config["data"].pop("index_paths", None)
    ds = _build_dataset(config, args.split)
    device = get_device(config.get("device", "auto"))
    model = _build_model(config, ds.dof, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    loader = DataLoader(ds, batch_size=int(config["data"].get("batch_size", 8)), shuffle=False, num_workers=0)

    report = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        report[mode] = _eval_mode(model, loader, device, mode)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
