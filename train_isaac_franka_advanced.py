from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.renderer import save_tensor_image
from src.robotics.franka_advanced_loss import SequenceLossWeights, integrate_deltas, sequence_prediction_loss
from src.robotics.franka_kinematics import FrankaKinematicPrior
from src.robotics.franka_temporal_policy import TemporalVisionFrankaPolicy
from src.robotics.isaac_franka_sequence_dataset import IsaacFrankaSequenceDataset
from src.utils import count_parameters, configure_torch_runtime, ensure_dir, get_device, load_config, save_config, set_seed


def _as_index_paths(config: Dict[str, Any]) -> Any:
    data_cfg = config["data"]
    if "index_paths" in data_cfg and data_cfg["index_paths"]:
        return data_cfg["index_paths"]
    return data_cfg["index_path"]


def build_dataloaders(config: Dict[str, Any]) -> tuple[DataLoader, DataLoader, int]:
    data_cfg = config["data"]
    common = dict(
        index_paths=_as_index_paths(config),
        image_size=int(data_cfg.get("image_size", 128)),
        context_len=int(data_cfg.get("context_len", 4)),
        horizon=int(data_cfg.get("horizon", 8)),
        stride=int(data_cfg.get("stride", 1)),
        val_fraction=float(data_cfg.get("val_fraction", 0.2)),
        split_mode=str(data_cfg.get("split_mode", "temporal")),
        max_windows=data_cfg.get("max_windows"),
        cache_images=bool(data_cfg.get("cache_images", False)),
        require_contiguous_steps=bool(data_cfg.get("require_contiguous_steps", False)),
        expected_step_delta=data_cfg.get("expected_step_delta"),
        max_ee_step=data_cfg.get("max_ee_step"),
        max_q_step=data_cfg.get("max_q_step"),
        image_preprocess=str(data_cfg.get("image_preprocess", "none")),
    )
    aug_cfg = dict(
        augment=bool(data_cfg.get("augment", False)),
        brightness_jitter=float(data_cfg.get("brightness_jitter", 0.0)),
        contrast_jitter=float(data_cfg.get("contrast_jitter", 0.0)),
        noise_std=float(data_cfg.get("noise_std", 0.0)),
        image_dropout_prob=float(data_cfg.get("image_dropout_prob", 0.0)),
    )
    train_set = IsaacFrankaSequenceDataset(split="train", **common, **aug_cfg)
    val_set = IsaacFrankaSequenceDataset(split="val", **common)
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, train_set.dof


def build_model(config: Dict[str, Any], dof: int, device: torch.device) -> TemporalVisionFrankaPolicy:
    model_cfg = config["model"]
    data_cfg = config["data"]
    model = TemporalVisionFrankaPolicy(
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
    )
    return model.to(device)


def build_loss_weights(config: Dict[str, Any]) -> SequenceLossWeights:
    cfg = config.get("loss", {})
    return SequenceLossWeights(
        delta=float(cfg.get("delta", 1.0)),
        q=float(cfg.get("q", 0.5)),
        ee=float(cfg.get("ee", 10.0)),
        terminal_ee=float(cfg.get("terminal_ee", 3.0)),
        smooth=float(cfg.get("smooth", 0.05)),
        joint_limit=float(cfg.get("joint_limit", 0.02)),
        fk_consistency=float(cfg.get("fk_consistency", 0.0)),
        goal=float(cfg.get("goal", 0.0)),
    )


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_model(model: TemporalVisionFrankaPolicy, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(
        batch["frames"],
        batch["q_context"],
        batch["ee_context"],
        batch["cube_context"],
        batch["target_position"],
    )


def run_epoch(
    model: TemporalVisionFrankaPolicy,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    weights: SequenceLossWeights,
    grad_clip_norm: float,
    fk_module: FrankaKinematicPrior | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if fk_module is not None:
        fk_module.train(is_train)
    totals: Dict[str, float] = {}
    seen = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            pred = forward_model(model, batch)
            loss, metrics = sequence_prediction_loss(pred, batch, weights, fk_module=fk_module)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                params = list(model.parameters()) + ([] if fk_module is None else list(fk_module.parameters()))
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            optimizer.step()
        bs = int(batch["frames"].shape[0])
        seen += bs
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v) * bs
    return {k: v / max(seen, 1) for k, v in totals.items()}


def save_checkpoint(
    path: Path,
    model: TemporalVisionFrankaPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best: float,
    config: Dict[str, Any],
    fk_module: FrankaKinematicPrior | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_terminal_ee_dist": best,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "fk_state": None if fk_module is None else fk_module.state_dict(),
            "config": config,
        },
        path,
    )


def save_metrics(rows: list[Dict[str, float]], csv_path: Path, png_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    epochs = [r["epoch"] for r in rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, [r["train_terminal_ee_dist"] for r in rows], label="train terminal EE")
    plt.plot(epochs, [r["val_terminal_ee_dist"] for r in rows], label="val terminal EE")
    plt.plot(epochs, [r["val_ee_dist"] for r in rows], label="val mean EE", linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel("error, meters")
    plt.title("Advanced Isaac Franka temporal training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def save_demo(model: TemporalVisionFrankaPolicy, loader: DataLoader, device: torch.device, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    batch = next(iter(loader))
    batch_device = move_batch(batch, device)
    with torch.no_grad():
        pred = forward_model(model, batch_device)
        q_pred = integrate_deltas(batch_device["q0"], pred["delta_q_seq"])
    save_tensor_image(batch["frames"][0, -1], out_dir / "advanced_context_last_frame.png")

    true_ee = batch["future_ee_positions"][0].cpu()
    pred_ee = pred["ee_seq"][0].cpu()
    horizon = list(range(1, true_ee.shape[0] + 1))
    plt.figure(figsize=(7.5, 4.5))
    for axis, name in enumerate(["x", "y", "z"]):
        plt.plot(horizon, true_ee[:, axis], label=f"target {name}")
        plt.plot(horizon, pred_ee[:, axis], linestyle="--", label=f"pred {name}")
    plt.xlabel("future step")
    plt.ylabel("EE coordinate, meters")
    plt.title("Predicted vs target end-effector rollout")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "advanced_ee_rollout.png", dpi=160)
    plt.close()

    report = {
        "target_future_delta_q_first": batch["future_delta_q"][0, 0].tolist(),
        "pred_future_delta_q_first": pred["delta_q_seq"][0, 0].cpu().tolist(),
        "target_future_ee": true_ee.tolist(),
        "pred_future_ee": pred_ee.tolist(),
        "target_future_q_last": batch["future_joint_positions"][0, -1].tolist(),
        "pred_future_q_last": q_pred[0, -1].cpu().tolist(),
        "target_place_position": batch["target_position"][0].tolist(),
    }
    with (out_dir / "advanced_sample_prediction.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train advanced temporal Isaac Franka model.")
    parser.add_argument("--config", default="configs/isaac_franka_advanced.yaml")
    parser.add_argument("--index", default=None, help="Override data.index_path. For several files, separate with semicolons.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.index is not None:
        config.setdefault("data", {})["index_path"] = args.index
        config["data"].pop("index_paths", None)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.output_dir is not None:
        config.setdefault("project", {})["output_dir"] = args.output_dir

    configure_torch_runtime(config.get("runtime", {}).get("num_threads"))
    set_seed(int(config.get("seed", 23)))
    device = get_device(config.get("device", "auto"))
    out_dir = ensure_dir(config["project"]["output_dir"])
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    save_config(config, out_dir / "config.yaml")

    train_loader, val_loader, dof = build_dataloaders(config)
    model = build_model(config, dof=dof, device=device)
    loss_weights = build_loss_weights(config)
    fk_module = None
    if loss_weights.fk_consistency > 0:
        fk_module = FrankaKinematicPrior(learn_calibration=True).to(device)

    params = list(model.parameters()) + ([] if fk_module is None else list(fk_module.parameters()))
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config["training"].get("lr", 3e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["training"].get("epochs", 80))),
        eta_min=float(config["training"].get("min_lr", 1e-5)),
    )

    print(f"Device: {device}")
    print(
        f"Temporal Isaac windows: train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"dof={dof} context={config['data'].get('context_len')} horizon={config['data'].get('horizon')}"
    )
    if hasattr(train_loader.dataset, "skipped_windows"):
        print(f"Skipped invalid windows: {train_loader.dataset.skipped_windows}")
    print(f"Trainable parameters: {count_parameters(model) + (0 if fk_module is None else count_parameters(fk_module)):,}")

    rows: list[Dict[str, float]] = []
    best = float("inf")
    no_improve = 0
    patience = int(config["training"].get("early_stop_patience", 20))
    for epoch in range(1, int(config["training"].get("epochs", 80)) + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            weights=loss_weights,
            grad_clip_norm=float(config["training"].get("grad_clip_norm", 1.0)),
            fk_module=fk_module,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            weights=loss_weights,
            grad_clip_norm=0.0,
            fk_module=fk_module,
        )
        scheduler.step()
        row: Dict[str, float] = {"epoch": float(epoch), "lr": float(scheduler.get_last_lr()[0])}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"val_{k}": v for k, v in val_stats.items()})
        rows.append(row)
        print(
            f"epoch={epoch:03d} train_term_ee={row['train_terminal_ee_dist']:.4f}m "
            f"val_term_ee={row['val_terminal_ee_dist']:.4f}m val_mean_ee={row['val_ee_dist']:.4f}m "
            f"val_q_abs={row['val_q_abs']:.5f}"
        )
        score = val_stats["terminal_ee_dist"]
        if score < best:
            best = score
            no_improve = 0
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best, config, fk_module=fk_module)
        else:
            no_improve += 1
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, best, config, fk_module=fk_module)
        if patience > 0 and no_improve >= patience:
            print(f"Early stopping: no validation improvement for {patience} epochs.")
            break

    save_metrics(rows, out_dir / "isaac_franka_advanced_metrics.csv", out_dir / "isaac_franka_advanced_loss_curve.png")
    save_demo(model, val_loader, device, out_dir / "plots")
    print(f"Best val terminal EE distance: {best:.4f} m")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
