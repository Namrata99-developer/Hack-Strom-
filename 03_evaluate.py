from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hackathon_seg.config import dataset_split_dir, load_config
from hackathon_seg.constants import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from hackathon_seg.data import SegmentationDataset, build_eval_transform, train_id_to_raw_mask
from hackathon_seg.inference import predict_logits_with_tta
from hackathon_seg.metrics import batch_confusion_matrix, metrics_from_confusion_matrix
from hackathon_seg.models import build_model
from hackathon_seg.utils import colorize_mask, ensure_dir, print_metrics, save_comparison_figure, save_json


def restore_mask_to_original_size(mask: np.ndarray, cfg: dict, original_height: int, original_width: int) -> np.ndarray:
    resized_height = int(cfg["inference"]["image_height"])
    resized_width = int(cfg["inference"]["image_width"])
    mask = mask[:resized_height, :resized_width]
    if mask.shape[0] == original_height and mask.shape[1] == original_width:
        return mask
    return cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)


def compute_single_image_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    pred_tensor = torch.from_numpy(pred_mask.astype(np.int64))
    gt_tensor = torch.from_numpy(gt_mask.astype(np.int64))
    valid_mask = gt_tensor != IGNORE_INDEX

    total_valid_pixels = int(valid_mask.sum().item())
    matched_pixels = int(((pred_tensor == gt_tensor) & valid_mask).sum().item())
    pixel_accuracy = (matched_pixels / total_valid_pixels) if total_valid_pixels else float("nan")

    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    valid_gt = gt_tensor[valid_mask]
    valid_pred = pred_tensor[valid_mask]
    indices = valid_gt * NUM_CLASSES + valid_pred
    confusion += torch.bincount(indices, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    metrics = metrics_from_confusion_matrix(confusion)

    row = {
        "mean_iou": float(metrics["mean_iou"]),
        "pixel_accuracy": float(pixel_accuracy),
        "matched_pixels": matched_pixels,
        "total_valid_pixels": total_valid_pixels,
    }
    for class_name, class_iou in zip(CLASS_NAMES, metrics["per_class_iou"]):
        row[f"iou_{class_name.lower().replace(' ', '_')}"] = float(class_iou)
    return row


def save_text_metrics(metrics: dict, output_dir: Path) -> None:
    lines = [
        "EVALUATION RESULTS",
        "=" * 40,
        f"Mean IoU: {metrics['mean_iou']:.4f}",
        f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}",
        "",
        "Per-class IoU:",
    ]
    for class_name, class_iou in zip(CLASS_NAMES, metrics["per_class_iou"]):
        value = "N/A" if np.isnan(class_iou) else f"{class_iou:.4f}"
        lines.append(f"  {class_name:<18} {value}")
    (output_dir / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")


def save_metric_tables(metrics: dict, confusion: torch.Tensor, output_dir: Path) -> None:
    per_class_df = pd.DataFrame(
        {
            "class_name": CLASS_NAMES,
            "iou": metrics["per_class_iou"],
        }
    )
    per_class_df.to_csv(output_dir / "per_class_iou.csv", index=False)

    confusion_df = pd.DataFrame(confusion.numpy(), index=CLASS_NAMES, columns=CLASS_NAMES)
    confusion_df.to_csv(output_dir / "confusion_matrix.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on val or test.")
    parser.add_argument("--config", default=str(CURRENT_DIR / "configs" / "exp3_final_last_run.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(cfg["train"]["save_dir"]) / "best.ckpt"

    split_dir = dataset_split_dir(cfg, args.split)
    output_dir = ensure_dir(Path(cfg["train"]["save_dir"]) / f"eval_{args.split}")
    raw_dir = ensure_dir(output_dir / "predictions_raw")
    color_dir = ensure_dir(output_dir / "predictions_color")
    comparison_dir = ensure_dir(output_dir / "comparisons")

    dataset = SegmentationDataset(split_dir, build_eval_transform(cfg), with_masks=True)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["inference"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["inference"]["num_workers"]),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    saved_samples = 0
    per_image_rows = []
    max_comparisons = int(cfg["inference"].get("save_comparisons", 12))
    save_all_comparisons = max_comparisons <= 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            if cfg["inference"]["use_tta"]:
                logits = predict_logits_with_tta(
                    model,
                    images,
                    scales=list(cfg["inference"]["scales"]),
                    hflip=bool(cfg["inference"]["hflip_tta"]),
                )
            else:
                logits = model(images)

            confusion += batch_confusion_matrix(logits, masks, NUM_CLASSES, IGNORE_INDEX)
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
            gt_masks = masks.detach().cpu().numpy()

            for index, name in enumerate(batch["name"]):
                original_height = int(batch["original_height"][index])
                original_width = int(batch["original_width"][index])

                pred_train_mask = restore_mask_to_original_size(
                    predictions[index].astype(np.uint8),
                    cfg,
                    original_height,
                    original_width,
                )
                pred_raw_mask = train_id_to_raw_mask(pred_train_mask)
                pred_color = colorize_mask(pred_train_mask)
                gt_train_mask = restore_mask_to_original_size(
                    gt_masks[index].astype(np.uint8),
                    cfg,
                    original_height,
                    original_width,
                )

                cv2.imwrite(str(raw_dir / name), pred_raw_mask)
                cv2.imwrite(str(color_dir / name), cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                per_image_row = {
                    "name": name,
                    "original_height": original_height,
                    "original_width": original_width,
                }
                per_image_row.update(compute_single_image_metrics(pred_train_mask, gt_train_mask))
                per_image_rows.append(per_image_row)

                if save_all_comparisons or saved_samples < max_comparisons:
                    save_comparison_figure(
                        image_tensor=batch["image"][index],
                        pred_mask=pred_train_mask,
                        gt_mask=gt_train_mask,
                        output_path=comparison_dir / f"{Path(name).stem}_comparison.png",
                        title=name,
                    )
                    saved_samples += 1

    metrics = metrics_from_confusion_matrix(confusion)
    print_metrics(metrics)
    save_json(
        {
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "mean_iou": metrics["mean_iou"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "per_class_iou": metrics["per_class_iou"].tolist(),
        },
        output_dir / "metrics.json",
    )
    save_text_metrics(metrics, output_dir)
    save_metric_tables(metrics, confusion, output_dir)
    pd.DataFrame(per_image_rows).to_csv(output_dir / "per_image_scores.csv", index=False)
    print(f"\nSaved predictions and metrics to {output_dir}")


if __name__ == "__main__":
    main()
