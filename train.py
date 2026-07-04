#!/usr/bin/env python3
"""Training entry point.

Usage
-----
    python train.py --task seg --model unet --data_root ./brisc2025 [options]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from src.config import Config, DataConfig, OptimizerConfig, PathConfig, TrainingConfig
from src.data.dataset import BriscDataset
from src.models import create_model
from src.training.trainer import Trainer
from src.utils.logging import get_logger
from src.utils.misc import set_seed

logger = get_logger(__name__)


def build_config(args: argparse.Namespace) -> Config:
    """Build a ``Config`` dataclass from CLI arguments."""
    return Config(
        data=DataConfig(
            data_root=args.data_root,
            img_size=args.img_size,
            batch_size=args.batch,
            num_workers=args.num_workers,
        ),
        optim=OptimizerConfig(
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
        ),
        training=TrainingConfig(
            task=args.task,
            model=args.model,
            epochs=args.epochs,
            use_amp=not args.no_amp,
            seed=args.seed,
        ),
        paths=PathConfig(
            checkpoint_dir=args.checkpoint_dir,
            report_dir=args.report_dir,
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentation / classification model")
    parser.add_argument("--task", type=str, choices=["seg", "cls", "joint"], required=True)
    parser.add_argument("--model", type=str, choices=["unet", "attunet", "bifpn"], required=True)
    parser.add_argument("--data_root", type=str, required=True, help="Path to BRISC2025 dataset")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8, help="Batch size (physical)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="runs")
    parser.add_argument("--report_dir", type=str, default="reports")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    cfg.paths.ensure_dirs()
    set_seed(cfg.training.seed)

    device = torch.device(cfg.device)
    logger.info("Using device: %s", device)
    logger.info("Config: %s", cfg)

    # Datasets
    train_ds = BriscDataset(
        cfg.data.data_root, split="train",
        task=cfg.training.task, img_size=cfg.data.img_size,
    )
    val_ds = BriscDataset(
        cfg.data.data_root, split="test",
        task=cfg.training.task, img_size=cfg.data.img_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
    )

    # Model
    model = create_model(
        cfg.training.model,
        num_classes=1,
        num_cls_labels=4,
    )

    # Train
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.fit()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
