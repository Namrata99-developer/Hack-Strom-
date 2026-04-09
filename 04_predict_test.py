from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hackathon_seg.config import dataset_split_dir, load_config
from hackathon_seg.data import SegmentationDataset, build_eval_transform, train_id_to_raw_mask
from hackathon_seg.inference import predict_logits_with_tta
from hackathon_seg.models import build_model
from hackathon_seg.utils import colorize_mask, ensure_dir


def restore_mask_to_original_size(mask, cfg: dict, original_height: int, original_width: int):
    resized_height = int(cfg["inference"]["image_height"])
    resized_width = int(cfg["inference"]["image_width"])
    mask = mask[:resized_height, :resized_width]
    if mask.shape[0] == original_height and mask.shape[1] == original_width:
        return mask
    return cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw-ID predictions for a test folder.")
    parser.add_argument("--config", default=str(CURRENT_DIR / "configs" / "exp3_final_last_run.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split_dir", default=None, help="Optional override path with Color_Images/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_dir = Path(args.split_dir) if args.split_dir else dataset_split_dir(cfg, "test")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(cfg["train"]["save_dir"]) / "best.ckpt"

    output_dir = ensure_dir(Path(cfg["train"]["save_dir"]) / "submission_predictions")
    raw_dir = ensure_dir(output_dir / "raw_ids")
    color_dir = ensure_dir(output_dir / "colorized")

    dataset = SegmentationDataset(split_dir, build_eval_transform(cfg), with_masks=False)
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

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            images = batch["image"].to(device, non_blocking=True)
            if cfg["inference"]["use_tta"]:
                logits = predict_logits_with_tta(
                    model,
                    images,
                    scales=list(cfg["inference"]["scales"]),
                    hflip=bool(cfg["inference"]["hflip_tta"]),
                )
            else:
                logits = model(images)

            predictions = logits.argmax(dim=1).detach().cpu().numpy()

            for index, name in enumerate(batch["name"]):
                original_height = int(batch["original_height"][index])
                original_width = int(batch["original_width"][index])
                pred_train_mask = restore_mask_to_original_size(
                    predictions[index].astype("uint8"),
                    cfg,
                    original_height,
                    original_width,
                )
                pred_raw_mask = train_id_to_raw_mask(pred_train_mask)
                pred_color = colorize_mask(pred_train_mask)

                cv2.imwrite(str(raw_dir / name), pred_raw_mask)
                cv2.imwrite(str(color_dir / name), cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

    print(f"Saved raw-ID predictions to {raw_dir}")
    print(f"Saved color predictions to {color_dir}")


if __name__ == "__main__":
    main()
