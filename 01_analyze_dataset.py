from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from hackathon_seg.config import load_config
from hackathon_seg.data import compute_class_pixel_counts
from hackathon_seg.utils import ensure_dir, save_json


def print_split_stats(split_name: str, stats: dict) -> None:
    print(f"\n=== {split_name.upper()} ===")
    print(f"Mask files: {stats['mask_files']}")
    print(f"Unique raw IDs: {stats['raw_unique_values']}")
    for item in stats["classes"]:
        print(
            f"  {item['raw_id']:>5} {item['class_name']:<15} "
            f"{item['pixel_fraction'] * 100:>7.3f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze class distribution for each split.")
    parser.add_argument("--config", default=str(CURRENT_DIR / "configs" / "exp3_final_last_run.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = Path(cfg["dataset"]["root_dir"])
    output_dir = ensure_dir(Path(cfg["train"]["save_dir"]) / "analysis")

    report = {}
    for split_key, split_dir_name in [
        ("train", cfg["dataset"]["train_dir"]),
        ("val", cfg["dataset"]["val_dir"]),
        ("test", cfg["dataset"]["test_dir"]),
    ]:
        stats = compute_class_pixel_counts(dataset_root / split_dir_name)
        report[split_key] = stats
        print_split_stats(split_key, stats)

    save_json(report, output_dir / "dataset_stats.json")
    print(f"\nSaved dataset analysis to {output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
