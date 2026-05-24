from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.dataset import SyntheticPhysicsDataset
from src.losses import property_mae, trajectory_mse
from src.physics_engine import DiffPhysicsEngine
from src.property_estimator import VisionPropertyEstimator
from src.renderer import save_tensor_image, save_trajectory_overlay
from src.utils import configure_torch_runtime, ensure_dir, get_device, load_config, save_json, set_seed


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


def plot_property_scatter(true_props: torch.Tensor, pred_props: torch.Tensor, out_dir: Path) -> None:
    names = ["mass", "restitution", "friction"]
    out_dir.mkdir(parents=True, exist_ok=True)
    true_np = true_props.numpy()
    pred_np = pred_props.numpy()
    for i, name in enumerate(names):
        plt.figure(figsize=(4, 4))
        plt.scatter(true_np[:, i], pred_np[:, i], s=14, alpha=0.75)
        lo = min(true_np[:, i].min(), pred_np[:, i].min())
        hi = max(true_np[:, i].max(), pred_np[:, i].max())
        plt.plot([lo, hi], [lo, hi], "--", linewidth=1)
        plt.xlabel(f"true {name}")
        plt.ylabel(f"predicted {name}")
        plt.title(f"{name}: predicted vs true")
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_{name}.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained vision-diff-physics checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    configure_torch_runtime(config.get("runtime", {}).get("num_threads"))
    checkpoint_path = Path(args.checkpoint or config["evaluation"]["checkpoint"])
    out_dir = ensure_dir(args.output_dir or Path(config["project"]["output_dir"]) / "eval")

    set_seed(int(config["seed"]) + 123)
    device = get_device(config.get("device", "auto"))

    dataset = SyntheticPhysicsDataset(
        num_samples=config["data"]["test_samples"],
        image_size=config["data"]["image_size"],
        horizon=config["simulation"]["horizon"],
        dt=config["simulation"]["dt"],
        seed=int(config["seed"]) + 20_000,
        prop_mins=config["properties"]["mins"],
        prop_maxs=config["properties"]["maxs"],
        action_scale=config["data"]["action_scale"],
        observation_noise_std=config["data"]["observation_noise_std"],
    )
    loader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=False)
    model, engine = build_model_and_engine(config, device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_pred_props = []
    all_true_props = []
    all_traj_losses = []
    overlay_saved = 0
    max_overlays = int(config["evaluation"].get("max_overlay_samples", 8))

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            pred_props = model(batch_gpu["frame"])
            pred_traj = engine(batch_gpu["init_pos"], batch_gpu["init_vel"], pred_props, batch_gpu["actions"])
            per_sample_traj = ((pred_traj - batch_gpu["observed_traj"]) ** 2).mean(dim=(1, 2))
            all_traj_losses.append(per_sample_traj.cpu())
            all_pred_props.append(pred_props.cpu())
            all_true_props.append(batch["true_props"].cpu())

            for j in range(batch["frame"].shape[0]):
                if overlay_saved >= max_overlays:
                    break
                idx = batch_index * config["data"]["batch_size"] + j
                sample_dir = out_dir / "overlays"
                save_tensor_image(batch["frame"][j], sample_dir / f"sample_{idx:03d}_frame.png")
                save_trajectory_overlay(
                    observed=batch["observed_traj"][j],
                    predicted=pred_traj[j].cpu(),
                    frame=batch["frame"][j],
                    path=sample_dir / f"sample_{idx:03d}_trajectory.png",
                    title=f"sample {idx:03d}",
                )
                overlay_saved += 1

    pred_props = torch.cat(all_pred_props, dim=0)
    true_props = torch.cat(all_true_props, dim=0)
    traj_losses = torch.cat(all_traj_losses, dim=0)
    mae = property_mae(pred_props, true_props)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "traj_mse_mean": float(traj_losses.mean()),
        "traj_mse_median": float(traj_losses.median()),
        "mass_mae": float(mae[0]),
        "restitution_mae": float(mae[1]),
        "friction_mae": float(mae[2]),
    }
    save_json(metrics, out_dir / "metrics.json")
    plot_property_scatter(true_props, pred_props, out_dir / "plots")

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_mass", "true_restitution", "true_friction", "pred_mass", "pred_restitution", "pred_friction"])
        for t, p in zip(true_props.tolist(), pred_props.tolist()):
            writer.writerow(t + p)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Saved evaluation artifacts to {out_dir}")


if __name__ == "__main__":
    main()
