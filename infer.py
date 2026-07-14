#!/usr/bin/env python3
"""Inference / evaluation entry point.
Usage
-----
    python infer.py --task joint --model unet --data_root ./brisc2025 \
        --ckpt runs/joint_unet/best.ckpt [options]
"""
from __future__ import annotations
import argparse
from src.config import Config, DataConfig, PathConfig, TrainingConfig
from src.inference.predictor import Predictor
from src.utils.logging import get_logger
logger = get_logger(__name__)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument("--task", type=str, choices=["seg", "cls", "joint"], required=True)
    parser.add_argument("--model", type=str, choices=["unet", "attunet", "bifpn"], required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--report_dir", type=str, default="reports")
    return parser.parse_args(argv)
def main() -> None:
    args = parse_args()
    cfg = Config(
        data=DataConfig(
            data_root=args.data_root,
            img_size=args.size,
            batch_size=args.batch,
            num_workers=args.num_workers,
        ),
        training=TrainingConfig(task=args.task, model=args.model),
        paths=PathConfig(
            output_dir=args.output_dir,
            report_dir=args.report_dir,
        ),
    )
    # Pass --ckpt directly to Predictor; it overrides the default config path
    predictor = Predictor(cfg, ckpt_path=args.ckpt)
    metrics = predictor.run()
    logger.info("Final metrics: %s", metrics)
if __name__ == "__main__":
    main()
