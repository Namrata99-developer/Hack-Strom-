from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from hackathon_seg.config import dataset_split_dir, load_config
from hackathon_seg.constants import IGNORE_INDEX, NUM_CLASSES
from hackathon_seg.data import (
    SegmentationDataset,
    build_eval_transform,
    build_train_transform,
    compute_image_sample_weights,
    compute_class_pixel_counts,
    make_enet_class_weights,
)
from hackathon_seg.losses import CombinedSegmentationLoss
from hackathon_seg.metrics import batch_confusion_matrix, summarize_metrics
from hackathon_seg.models import build_model
from hackathon_seg.utils import ensure_dir, load_history, print_metrics, save_history, save_json, set_seed


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    grad_clip_norm: float,
) -> dict:
    is_training = optimizer is not None
    model.train(is_training)

    losses = []
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    progress = tqdm(loader, leave=False, desc="train" if is_training else "eval")

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        confusion += batch_confusion_matrix(logits.detach(), masks, NUM_CLASSES, IGNORE_INDEX)
        losses.append(float(loss.item()))
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return summarize_metrics(losses, confusion)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_iou: float,
    epochs_without_improvement: int,
    cfg: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_iou": best_iou,
            "epochs_without_improvement": epochs_without_improvement,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": cfg,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stronger segmentation model for Hackstorm.")
    parser.add_argument("--config", default=str(CURRENT_DIR / "configs" / "exp3_final_last_run.yaml"))
    parser.add_argument(
        "--resume",
        default="auto",
        help="Checkpoint path to resume from. Use 'auto' for save_dir/last.ckpt or 'none' to disable resume.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["train"]["seed"]))

    output_dir = ensure_dir(cfg["train"]["save_dir"])
    save_json(cfg, output_dir / "resolved_config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_stats = compute_class_pixel_counts(dataset_split_dir(cfg, "train"))
    save_json(train_stats, output_dir / "train_class_stats.json")

    train_ds = SegmentationDataset(dataset_split_dir(cfg, "train"), build_train_transform(cfg), with_masks=True)
    val_ds = SegmentationDataset(dataset_split_dir(cfg, "val"), build_eval_transform(cfg), with_masks=True)

    train_sampler = None
    train_shuffle = True
    if bool(cfg["train"].get("use_weighted_sampler", False)):
        sample_weights = compute_image_sample_weights(train_ds, train_stats, cfg)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
        save_json(
            {
                "min_weight": float(sample_weights.min().item()),
                "max_weight": float(sample_weights.max().item()),
                "mean_weight": float(sample_weights.mean().item()),
                "focus_raw_ids": [int(raw_id) for raw_id in cfg["train"].get("sampler_focus_raw_ids", [])],
            },
            output_dir / "train_sampler_stats.json",
        )
        print(
            "Using weighted sampler "
            f"(min={sample_weights.min().item():.3f}, max={sample_weights.max().item():.3f}, "
            f"mean={sample_weights.mean().item():.3f})"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["inference"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["inference"]["num_workers"]),
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    class_weights = None
    if cfg["train"]["use_class_weights"]:
        class_weights = make_enet_class_weights(train_stats, device)
        print(f"Using class weights: {class_weights.detach().cpu().numpy().round(3).tolist()}")

    criterion = CombinedSegmentationLoss(
        class_weights=class_weights,
        ce_weight=float(cfg["train"]["ce_weight"]),
        dice_weight=float(cfg["train"]["dice_weight"]),
        focal_weight=float(cfg["train"].get("focal_weight", 0.0)),
        focal_gamma=float(cfg["train"].get("focal_gamma", 2.0)),
        label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["train"]["epochs"]))
    scaler = GradScaler(device=device.type, enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    history = []
    best_iou = -1.0
    epochs_without_improvement = 0
    start_epoch = 1

    if args.resume.lower() == "auto":
        resume_path = output_dir / "last.ckpt"
    elif args.resume.lower() == "none":
        resume_path = None
    else:
        resume_path = Path(args.resume)

    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_iou = float(checkpoint.get("best_iou", -1.0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        history = load_history(output_dir)
        print(f"Resuming from checkpoint: {resume_path}")
        print(f"Resuming at epoch {start_epoch} with best val mIoU {best_iou:.4f}")

    if start_epoch > int(cfg["train"]["epochs"]):
        print("Training already completed for the configured number of epochs.")
        print(f"Artifacts saved to {output_dir}")
        return

    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            use_amp=bool(cfg["train"]["amp"]),
            grad_clip_norm=float(cfg["train"]["grad_clip_norm"]),
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scaler=scaler,
            criterion=criterion,
            device=device,
            use_amp=bool(cfg["train"]["amp"]),
            grad_clip_norm=0.0,
        )
        scheduler.step()

        epoch_row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_mean_iou": train_metrics["mean_iou"],
            "val_mean_iou": val_metrics["mean_iou"],
            "train_pixel_accuracy": train_metrics["pixel_accuracy"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
        }
        history.append(epoch_row)
        save_history(history, output_dir)

        print("Train metrics")
        print_metrics(train_metrics)
        print("Val metrics")
        print_metrics(val_metrics)

        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.ckpt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_iou,
                epochs_without_improvement,
                cfg,
            )
            save_json(
                {
                    "epoch": epoch,
                    "best_val_mean_iou": best_iou,
                    "per_class_iou": val_metrics["per_class_iou"].tolist(),
                },
                output_dir / "best_metrics.json",
            )
            print(f"Saved new best checkpoint with val mIoU {best_iou:.4f}")
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            output_dir / "last.ckpt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_iou,
            epochs_without_improvement,
            cfg,
        )

        if epochs_without_improvement >= int(cfg["train"]["patience"]):
            print("Early stopping triggered.")
            break

    print(f"\nTraining finished. Best validation mIoU: {best_iou:.4f}")
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
