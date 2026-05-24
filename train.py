from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.dataset import SyntheticPhysicsDataset
from src.optimizer import make_optimizer, make_scheduler, run_epoch
from src.physics_engine import DiffPhysicsEngine
from src.property_estimator import VisionPropertyEstimator
from src.renderer import save_tensor_image, save_trajectory_overlay
from src.utils import count_parameters, configure_torch_runtime, ensure_dir, get_device, load_config, save_config, set_seed


def build_dataloaders(config: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    sim_cfg = config["simulation"]
    prop_cfg = config["properties"]
    seed = int(config["seed"])

    train_set = SyntheticPhysicsDataset(
        num_samples=data_cfg["train_samples"],
        image_size=data_cfg["image_size"],
        horizon=sim_cfg["horizon"],
        dt=sim_cfg["dt"],
        seed=seed,
        prop_mins=prop_cfg["mins"],
        prop_maxs=prop_cfg["maxs"],
        action_scale=data_cfg["action_scale"],
        observation_noise_std=data_cfg["observation_noise_std"],
    )
    val_set = SyntheticPhysicsDataset(
        num_samples=data_cfg["val_samples"],
        image_size=data_cfg["image_size"],
        horizon=sim_cfg["horizon"],
        dt=sim_cfg["dt"],
        seed=seed + 10_000,
        prop_mins=prop_cfg["mins"],
        prop_maxs=prop_cfg["maxs"],
        action_scale=data_cfg["action_scale"],
        observation_noise_std=data_cfg["observation_noise_std"],
    )
    train_loader = DataLoader(
        train_set,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_model_and_engine(config: Dict[str, Any], device: torch.device) -> tuple[VisionPropertyEstimator, DiffPhysicsEngine]:
    prop_cfg = config["properties"]
    model_cfg = config["model"]
    sim_cfg = config["simulation"]

    model = VisionPropertyEstimator(
        image_channels=3,
        feature_dim=model_cfg["feature_dim"],
        base_channels=model_cfg["base_channels"],
        hidden_dim=model_cfg["hidden_dim"],
        prop_mins=prop_cfg["mins"],
        prop_maxs=prop_cfg["maxs"],
    ).to(device)
    engine = DiffPhysicsEngine(
        dt=sim_cfg["dt"],
        horizon=sim_cfg["horizon"],
        bounds=tuple(sim_cfg["bounds"]),
        radius=sim_cfg["radius"],
        friction_acceleration=sim_cfg["friction_acceleration"],
        linear_damping=sim_cfg["linear_damping"],
        velocity_epsilon=sim_cfg["velocity_epsilon"],
    ).to(device)
    return model, engine


def save_metrics_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in rows]
    train = [r["train_traj_mse"] for r in rows]
    val = [r["val_traj_mse"] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train, label="train trajectory MSE")
    plt.plot(epochs, val, label="val trajectory MSE")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Trajectory loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_checkpoint(
    path: Path,
    model: VisionPropertyEstimator,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    config: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_traj_mse": best_val,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def save_demo_predictions(
    model: VisionPropertyEstimator,
    engine: DiffPhysicsEngine,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        pred_props = model(batch_gpu["frame"])
        pred_traj = engine(
            batch_gpu["init_pos"], batch_gpu["init_vel"], pred_props, batch_gpu["actions"]
        )
    save_tensor_image(batch["frame"][0], out_dir / "input_frame.png")
    save_trajectory_overlay(
        observed=batch["observed_traj"][0],
        predicted=pred_traj[0].cpu(),
        frame=batch["frame"][0],
        path=out_dir / "trajectory_overlay.png",
        title="Validation trajectory: observed vs predicted",
    )
    comparison = torch.stack([batch_gpu["true_props"][0], pred_props[0]]).detach().cpu()
    labels = ["mass", "restitution", "friction"]
    fig, ax = plt.subplots(figsize=(6, 3))
    x = torch.arange(3).numpy()
    ax.bar(x - 0.18, comparison[0].numpy(), width=0.36, label="true")
    ax.bar(x + 0.18, comparison[1].numpy(), width=0.36, label="predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Physical property estimate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "property_bar.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vision-guided differentiable physics demo.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--output-dir", type=str, default=None, help="Override project.output_dir")
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
    model, engine = build_model_and_engine(config, device)
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    train_cfg = config["training"]
    optimizer = make_optimizer(model, train_cfg["lr"], train_cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, train_cfg["epochs"], train_cfg["min_lr"])

    rows: list[dict[str, float]] = []
    best_val = float("inf")
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_stats = run_epoch(
            model=model,
            engine=engine,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            property_aux_weight=train_cfg["property_aux_weight"],
            grad_clip_norm=train_cfg["grad_clip_norm"],
        )
        val_stats = run_epoch(
            model=model,
            engine=engine,
            loader=val_loader,
            optimizer=None,
            device=device,
            property_aux_weight=train_cfg["property_aux_weight"],
        )
        scheduler.step()

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        row.update(train_stats.as_dict("train"))
        row.update(val_stats.as_dict("val"))
        rows.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train traj {train_stats.traj_mse:.6f} | val traj {val_stats.traj_mse:.6f} | "
            f"val MAE mass/e/mu = {val_stats.mass_mae:.3f}/"
            f"{val_stats.restitution_mae:.3f}/{val_stats.friction_mae:.3f}"
        )

        if val_stats.traj_mse < best_val:
            best_val = val_stats.traj_mse
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best_val, config)
        if epoch % int(train_cfg["save_every"]) == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, best_val, config)

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, best_val, config)
        save_metrics_csv(rows, out_dir / "metrics.csv")
        plot_metrics(rows, out_dir / "loss_curve.png")

    save_demo_predictions(model, engine, val_loader, device, out_dir / "plots")
    print(f"Done. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
