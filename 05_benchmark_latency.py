from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hackathon_seg.config import dataset_split_dir, load_config
from hackathon_seg.data import SegmentationDataset, build_eval_transform
from hackathon_seg.inference import predict_logits_with_tta
from hackathon_seg.models import build_model
from hackathon_seg.utils import ensure_dir, save_json


def _resolve_input_dir(cfg: dict, split: str | None, split_dir: str | None) -> tuple[Path, str]:
    if split_dir:
        custom_path = Path(split_dir)
        return custom_path, custom_path.name or "custom"
    chosen_split = split or "test"
    return dataset_split_dir(cfg, chosen_split), chosen_split


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_latencies(latencies_ms: list[float], total_images: int) -> dict:
    if not latencies_ms:
        return {
            "images_benchmarked": 0,
            "latency_ms_mean": float("nan"),
            "latency_ms_median": float("nan"),
            "latency_ms_p90": float("nan"),
            "latency_ms_p95": float("nan"),
            "latency_ms_min": float("nan"),
            "latency_ms_max": float("nan"),
            "fps_mean": float("nan"),
        }

    sorted_vals = sorted(latencies_ms)

    def percentile(p: float) -> float:
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        index = min(len(sorted_vals) - 1, max(0, int(round((len(sorted_vals) - 1) * p))))
        return sorted_vals[index]

    mean_ms = statistics.mean(latencies_ms)
    return {
        "images_benchmarked": total_images,
        "latency_ms_mean": mean_ms,
        "latency_ms_median": statistics.median(latencies_ms),
        "latency_ms_p90": percentile(0.90),
        "latency_ms_p95": percentile(0.95),
        "latency_ms_min": min(latencies_ms),
        "latency_ms_max": max(latencies_ms),
        "fps_mean": (1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference latency and FPS for the v3 model.")
    parser.add_argument("--config", default=str(CURRENT_DIR / "configs" / "exp3_final_last_run.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--split_dir", default=None, help="Optional override path with Color_Images/ and optional Segmentation/")
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means use all batches")
    parser.add_argument("--disable_tta", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(cfg["train"]["save_dir"]) / "best.ckpt"
    input_dir, split_name = _resolve_input_dir(cfg, args.split, args.split_dir)

    with_masks = (input_dir / "Segmentation").exists()
    dataset = SegmentationDataset(input_dir, build_eval_transform(cfg), with_masks=with_masks)
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

    use_tta = bool(cfg["inference"]["use_tta"]) and not args.disable_tta
    benchmark_dir = ensure_dir(Path(cfg["train"]["save_dir"]) / f"benchmark_{split_name}")

    warmup_batches = max(0, int(args.warmup_batches))
    max_batches = int(args.max_batches)

    print(f"Using device: {device}")
    print(f"Benchmark input dir: {input_dir}")
    print(f"Using TTA: {use_tta}")
    print(f"With masks available: {with_masks}")

    latency_rows = []
    latencies_ms = []
    total_images = 0

    with torch.no_grad():
        iterator = tqdm(loader, desc=f"Benchmarking {split_name}")
        for batch_index, batch in enumerate(iterator):
            if max_batches > 0 and batch_index >= max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)

            _synchronize_if_needed(device)
            start_time = time.perf_counter()
            if use_tta:
                logits = predict_logits_with_tta(
                    model,
                    images,
                    scales=list(cfg["inference"]["scales"]),
                    hflip=bool(cfg["inference"]["hflip_tta"]),
                )
            else:
                logits = model(images)
            _ = logits.argmax(dim=1)
            _synchronize_if_needed(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            batch_size = int(images.shape[0])
            per_image_ms = elapsed_ms / max(batch_size, 1)

            if batch_index >= warmup_batches:
                latencies_ms.append(per_image_ms)
                total_images += batch_size
                for image_name in batch["name"]:
                    latency_rows.append(
                        {
                            "name": image_name,
                            "batch_index": batch_index,
                            "latency_ms_per_image": per_image_ms,
                            "tta_used": use_tta,
                        }
                    )

            display_ms = per_image_ms if batch_index >= warmup_batches else float("nan")
            iterator.set_postfix(lat_ms=f"{display_ms:.2f}" if display_ms == display_ms else "warmup")

    summary = _summarize_latencies(latencies_ms, total_images)
    summary.update(
        {
            "split": split_name,
            "input_dir": str(input_dir),
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "tta_used": use_tta,
            "warmup_batches": warmup_batches,
            "max_batches": max_batches,
        }
    )

    save_json(summary, benchmark_dir / "benchmark_summary.json")
    pd.DataFrame(latency_rows).to_csv(benchmark_dir / "per_image_latency.csv", index=False)

    print("\nBENCHMARK SUMMARY")
    print("=" * 40)
    print(f"Images benchmarked: {summary['images_benchmarked']}")
    print(f"Mean latency: {summary['latency_ms_mean']:.2f} ms/image")
    print(f"Median latency: {summary['latency_ms_median']:.2f} ms/image")
    print(f"P90 latency: {summary['latency_ms_p90']:.2f} ms/image")
    print(f"P95 latency: {summary['latency_ms_p95']:.2f} ms/image")
    print(f"Mean FPS: {summary['fps_mean']:.2f}")
    print(f"Saved benchmark outputs to {benchmark_dir}")


if __name__ == "__main__":
    main()
