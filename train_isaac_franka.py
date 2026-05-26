from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.robotics import IsaacFrankaEpisodeDataset, VisionFrankaPolicy
from src.renderer import save_tensor_image
from src.utils import count_parameters, configure_torch_runtime, ensure_dir, get_device, load_config, save_config, set_seed


def build_dataloaders(config: Dict[str, Any]) -> tuple[DataLoader, DataLoader, int]:
    data_cfg = config["data"]
    common = dict(
        index_path=data_cfg["index_path"],
        image_size=int(data_cfg.get("image_size", 128)),
        stride=int(data_cfg.get("stride", 1)),
        val_fraction=float(data_cfg.get("val_fraction", 0.2)),
        max_pairs=data_cfg.get("max_pairs"),
    )
    train_set = IsaacFrankaEpisodeDataset(split="train", **common)
    val_set = IsaacFrankaEpisodeDataset(split="val", **common)
    batch_size = int(data_cfg.get("batch_size", 12))
    num_workers = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, train_set.dof


def build_model(config: Dict[str, Any], dof: int, device: torch.device) -> VisionFrankaPolicy:
    cfg = config["model"]
    return VisionFrankaPolicy(
        dof=dof,
        feature_dim=int(cfg.get("feature_dim", 192)),
        base_channels=int(cfg.get("base_channels", 24)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        max_delta=float(cfg.get("max_delta", 0.25)),
    ).to(device)


def run_epoch(
    model: VisionFrankaPolicy,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    delta_weight: float,
    ee_weight: float,
    grad_clip_norm: float,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "delta_mse": 0.0, "ee_mse": 0.0, "ee_dist": 0.0, "delta_abs": 0.0}
    n = 0
    for batch in loader:
        frame = batch["frame"].to(device)
        q = batch["joint_positions"].to(device)
        ee = batch["end_effector_position"].to(device)
        cube = batch["cube_position"].to(device)
        target = batch["target_position"].to(device)
        delta_q = batch["delta_q"].to(device)
        next_ee = batch["next_end_effector_position"].to(device)

        with torch.set_grad_enabled(train):
            pred = model(frame, q, ee, cube, target)
            delta_mse = F.mse_loss(pred["delta_q"], delta_q)
            ee_mse = F.mse_loss(pred["next_ee"], next_ee)
            loss = float(delta_weight) * delta_mse + float(ee_weight) * ee_mse

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        bs = frame.shape[0]
        n += bs
        totals["loss"] += float(loss.detach().cpu()) * bs
        totals["delta_mse"] += float(delta_mse.detach().cpu()) * bs
        totals["ee_mse"] += float(ee_mse.detach().cpu()) * bs
        totals["ee_dist"] += float(torch.linalg.norm(pred["next_ee"].detach() - next_ee, dim=-1).mean().cpu()) * bs
        totals["delta_abs"] += float(torch.mean(torch.abs(pred["delta_q"].detach() - delta_q)).cpu()) * bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def save_checkpoint(path: Path, model: VisionFrankaPolicy, optimizer: torch.optim.Optimizer, epoch: int, best: float, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_ee_dist": best,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def save_metrics(rows: list[Dict[str, float]], csv_path: Path, png_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    epochs = [r["epoch"] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [r["train_ee_dist"] for r in rows], label="train EE dist")
    plt.plot(epochs, [r["val_ee_dist"] for r in rows], label="val EE dist")
    plt.xlabel("epoch")
    plt.ylabel("mean EE error, meters")
    plt.title("Isaac Franka imitation bridge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def save_demo(model: VisionFrankaPolicy, loader: DataLoader, device: torch.device, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        pred = model(
            batch["frame"].to(device),
            batch["joint_positions"].to(device),
            batch["end_effector_position"].to(device),
            batch["cube_position"].to(device),
            batch["target_position"].to(device),
        )
    save_tensor_image(batch["frame"][0], out_dir / "isaac_input_frame.png")
    report = {
        "target_delta_q": batch["delta_q"][0].tolist(),
        "pred_delta_q": pred["delta_q"][0].cpu().tolist(),
        "target_next_ee": batch["next_end_effector_position"][0].tolist(),
        "pred_next_ee": pred["next_ee"][0].cpu().tolist(),
        "target_place_position": batch["target_position"][0].tolist(),
    }
    import json

    with (out_dir / "sample_prediction.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RGB + state imitation model on collected Isaac Franka data.")
    parser.add_argument("--config", default="configs/isaac_franka.yaml")
    parser.add_argument("--index", default=None, help="Override data.index_path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.index is not None:
        config["data"]["index_path"] = args.index
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.output_dir is not None:
        config["project"]["output_dir"] = args.output_dir

    configure_torch_runtime(config.get("runtime", {}).get("num_threads"))
    set_seed(int(config.get("seed", 23)))
    device = get_device(config.get("device", "auto"))
    out_dir = ensure_dir(config["project"]["output_dir"])
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    save_config(config, out_dir / "config.yaml")

    train_loader, val_loader, dof = build_dataloaders(config)
    model = build_model(config, dof=dof, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 5e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["training"].get("epochs", 30))),
        eta_min=float(config["training"].get("min_lr", 1e-5)),
    )

    print(f"Device: {device}")
    print(f"Isaac records: train_pairs={len(train_loader.dataset)} val_pairs={len(val_loader.dataset)} dof={dof}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    rows: list[Dict[str, float]] = []
    best = float("inf")
    for epoch in range(1, int(config["training"].get("epochs", 30)) + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            delta_weight=float(config["training"].get("delta_weight", 1.0)),
            ee_weight=float(config["training"].get("ee_weight", 10.0)),
            grad_clip_norm=float(config["training"].get("grad_clip_norm", 1.0)),
        )
        val_stats = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            delta_weight=float(config["training"].get("delta_weight", 1.0)),
            ee_weight=float(config["training"].get("ee_weight", 10.0)),
            grad_clip_norm=0.0,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": train_stats["loss"],
            "train_delta_mse": train_stats["delta_mse"],
            "train_ee_mse": train_stats["ee_mse"],
            "train_ee_dist": train_stats["ee_dist"],
            "train_delta_abs": train_stats["delta_abs"],
            "val_loss": val_stats["loss"],
            "val_delta_mse": val_stats["delta_mse"],
            "val_ee_mse": val_stats["ee_mse"],
            "val_ee_dist": val_stats["ee_dist"],
            "val_delta_abs": val_stats["delta_abs"],
        }
        rows.append(row)
        print(
            f"epoch={epoch:03d} train_ee={row['train_ee_dist']:.4f}m "
            f"val_ee={row['val_ee_dist']:.4f}m val_delta_abs={row['val_delta_abs']:.5f}"
        )
        if val_stats["ee_dist"] < best:
            best = val_stats["ee_dist"]
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best, config)
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, best, config)

    save_metrics(rows, out_dir / "isaac_franka_metrics.csv", out_dir / "isaac_franka_loss_curve.png")
    save_demo(model, val_loader, device, out_dir / "plots")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
