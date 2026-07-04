#!/usr/bin/env python3
"""Visualisation entry point — compare separate vs. joint models.

Usage
-----
    python visualize.py [--index 42] [--models_root runs] [--data_root ./brisc2025]
"""

from __future__ import annotations

import argparse
import os

from src.utils.visualization import visualize_models
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise model predictions (separate vs. joint)"
    )
    parser.add_argument("--data_root", type=str, default="./brisc2025")
    parser.add_argument("--models_root", type=str, default="runs")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--index", type=int, default=None,
                        help="File index (0-859). If omitted, prompts interactively.")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    img_dir = os.path.join(args.data_root, "segmentation_task", "test", "images")
    mask_dir = os.path.join(args.data_root, "segmentation_task", "test", "masks")
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    max_index = min(859, len(img_files) - 1)

    if args.index is not None:
        index = args.index
    else:
        index = int(input(f"Enter an index (0 to {max_index}): "))

    if index < 0 or index > max_index:
        logger.error("Index out of range! Please enter between 0 and %d.", max_index)
        return

    img_name = img_files[index]
    mask_name = mask_files[index]
    img_path = os.path.join(img_dir, img_name)
    mask_path = os.path.join(mask_dir, mask_name)
    save_path = os.path.join(args.save_dir, f"compare_idx{index}.png")
    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("Visualizing index %d: %s", index, img_name)

    visualize_models(
        image_path=img_path,
        mask_path=mask_path,
        models_root=args.models_root,
        device=args.device,
        img_size=args.size,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
