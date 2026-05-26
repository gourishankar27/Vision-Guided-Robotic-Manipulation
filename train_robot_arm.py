from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.robotics import DifferentiablePlanarArm, SyntheticRobotReachDataset, VisionArmPolicy
from src.robotics.plotting import save_arm_reach_plot
from src.renderer import save_tensor_image
from src.utils import count_parameters, configure_torch_runtime, ensure_dir, get_device, load_config, save_config, set_seed


def build_dataloaders(config: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    robot_cfg = config["robot"]
    seed = int(config["seed"])

    train_set = SyntheticRobotReachDataset(
        num_samples=data_cfg["train_samples"],
        image_size=data_cfg["image_size"],
        link_lengths=robot_cfg["link_lengths"],
        seed=seed,
        noise_std=data_cfg.get("image_noise_std", 0.01),
        pre_render=True,
    )
    val_set = SyntheticRobotReachDataset(
        num_samples=data_cfg["val_samples"],
        image_size=data_cfg["image_size"],
        link_lengths=robot_cfg["link_lengths"],
        seed=seed + 10_000,
        noise_std=data_cfg.get("image_noise_std", 0.01),
        pre_render=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_model_and_engine(config: Dict[str, Any], device: torch.device) -> tuple[VisionArmPolicy, DifferentiablePlanarArm]:
    robot_cfg = config["robot"]
    model_cfg = config["model"]
    arm = DifferentiablePlanarArm(
        link_lengths=robot_cfg["link_lengths"],
        joint_limit=robot_cfg.get("joint_limit", 2.85),
        max_delta=robot_cfg.get("max_delta", 1.75),
        rollout_steps=robot_cfg.get("rollout_steps", 20),
    ).to(device)
    model = VisionArmPolicy(
        dof=arm.dof,
        image_channels=3,
        feature_dim=model_cfg.get("feature_dim", 128),
        base_channels=model_cfg.get("base_channels", 16),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        max_delta=robot_cfg.get("max_delta", 1.75),
    ).to(device)
    return model, arm


def run_epoch(
    model: VisionArmPolicy,
    arm: DifferentiablePlanarArm,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    imitation_weight: float,
    smoothness_weight: float,
    grad_clip_norm: float,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "ee_mse": 0.0, "action_mse": 0.0, "final_dist": 0.0}
    n = 0
    for batch in loader:
        frame = batch["frame"].to(device)
        q0 = batch["joint_angles"].to(device)
        target_xy = batch["target_xy"].to(device)
        oracle_delta = batch["oracle_delta_q"].to(device)

        with torch.set_grad_enabled(train):
            pred_delta = model(frame, q0)
            pred_ee = arm.final_end_effector(q0, pred_delta)
            ee_mse = F.mse_loss(pred_ee, target_xy)
            action_mse = F.mse_loss(pred_delta, oracle_delta)
            path = arm.rollout(q0, pred_delta)
            step_diff = path[:, 1:] - path[:, :-1]
            smoothness = (step_diff[:, 1:] - step_diff[:, :-1]).pow(2).mean() if path.shape[1] > 2 else torch.zeros((), device=device)
            loss = ee_mse + imitation_weight * action_mse + smoothness_weight * smoothness

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_size = frame.shape[0]
        n += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["ee_mse"] += float(ee_mse.detach().cpu()) * batch_size
        totals["action_mse"] += float(action_mse.detach().cpu()) * batch_size
        totals["final_dist"] += float(torch.linalg.norm(pred_ee.detach() - target_xy, dim=-1).mean().cpu()) * batch_size

    return {k: v / max(n, 1) for k, v in totals.items()}


def save_metrics(rows: list[Dict[str, float]], out_csv: Path, out_png: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    epochs = [r["epoch"] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [r["train_ee_mse"] for r in rows], label="train EE MSE")
    plt.plot(epochs, [r["val_ee_mse"] for r in rows], label="val EE MSE")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("end-effector MSE")
    plt.title("Vision-guided robot reaching")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_checkpoint(path: Path, model: VisionArmPolicy, optimizer: torch.optim.Optimizer, epoch: int, best_val: float, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_ee_mse": best_val,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def save_demo(model: VisionArmPolicy, loader: DataLoader, device: torch.device, out_dir: Path, link_lengths: list[float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        frame = batch["frame"].to(device)
        q0 = batch["joint_angles"].to(device)
        pred_delta = model(frame, q0).cpu()
    save_tensor_image(batch["frame"][0], out_dir / "robot_input_frame.png")
    save_arm_reach_plot(
        initial_q=batch["joint_angles"][0],
        predicted_delta=pred_delta[0],
        target_xy=batch["target_xy"][0],
        path=out_dir / "robot_reach_overlay.png",
        link_lengths=link_lengths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the vision-guided robotic-arm proxy.")
    parser.add_argument("--config", default="configs/robot_arm.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    configure_torch_runtime(config.get("runtime", {}).get("num_threads"))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.output_dir is not None:
        config["project"]["output_dir"] = args.output_dir

    set_seed(int(config["seed"]))
    device = get_device(config.get("device", "auto"))
    out_dir = ensure_dir(config["project"]["output_dir"])
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    save_config(config, out_dir / "config.yaml")

    train_loader, val_loader = build_dataloaders(config)
    model, arm = build_model_and_engine(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["training"]["epochs"])),
        eta_min=float(config["training"].get("min_lr", 1e-5)),
    )

    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    rows: list[Dict[str, float]] = []
    best = float("inf")
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_stats = run_epoch(
            model,
            arm,
            train_loader,
            device,
            optimizer,
            imitation_weight=float(config["training"].get("imitation_weight", 0.05)),
            smoothness_weight=float(config["training"].get("smoothness_weight", 0.001)),
            grad_clip_norm=float(config["training"].get("grad_clip_norm", 1.0)),
        )
        val_stats = run_epoch(
            model,
            arm,
            val_loader,
            device,
            optimizer=None,
            imitation_weight=float(config["training"].get("imitation_weight", 0.05)),
            smoothness_weight=float(config["training"].get("smoothness_weight", 0.001)),
            grad_clip_norm=0.0,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": train_stats["loss"],
            "train_ee_mse": train_stats["ee_mse"],
            "train_action_mse": train_stats["action_mse"],
            "train_final_dist": train_stats["final_dist"],
            "val_loss": val_stats["loss"],
            "val_ee_mse": val_stats["ee_mse"],
            "val_action_mse": val_stats["action_mse"],
            "val_final_dist": val_stats["final_dist"],
        }
        rows.append(row)
        print(
            f"epoch={epoch:03d} train_ee={row['train_ee_mse']:.6f} "
            f"val_ee={row['val_ee_mse']:.6f} val_dist={row['val_final_dist']:.4f}"
        )
        if val_stats["ee_mse"] < best:
            best = val_stats["ee_mse"]
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best, config)
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, best, config)

    save_metrics(rows, out_dir / "robot_metrics.csv", out_dir / "robot_loss_curve.png")
    save_demo(model, val_loader, device, out_dir / "plots", config["robot"]["link_lengths"])
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
