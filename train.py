#!/usr/bin/env python3
"""Training entry point.
Usage
-----
    python train.py --task seg --model unet --data_root ./brisc2025 [options]
"""
from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader
from src.config import (
    NUM_CLASSES_CLS,
    NUM_CLASSES_SEG,
    Config,
    DataConfig,
    OptimizerConfig,
    PathConfig,
    TrainingConfig,
)
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
            use_compile=args.compile,
            seed=args.seed,
            n_folds=args.n_folds,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
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
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile() for speedup")
    parser.add_argument("--n_folds", type=int, default=0, help="Number of K-fold CV folds (0=disabled)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases tracking")
    parser.add_argument("--wandb_project", type=str, default="brisc-tumor", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default="", help="WandB run name")
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
    n_folds = cfg.training.n_folds
    if n_folds > 1:
        # ── K-fold cross-validation ────────────────────────────────
        from sklearn.model_selection import StratifiedKFold
        full_ds = BriscDataset(
            cfg.data.data_root, split="train",
            task=cfg.training.task, img_size=cfg.data.img_size,
        )
        # Get labels for stratified split
        if cfg.training.task == "cls":
            labels = [label for _, label in full_ds.img_files]
        else:
            # For seg/joint, derive labels from filenames for stratification
            from src.config import FILENAME_CLASS_MAP
            labels = []
            for fname in full_ds.img_files:
                label = 0
                for key, cls_idx in FILENAME_CLASS_MAP.items():
                    if f"_{key}_" in fname.lower():
                        label = cls_idx
                        break
                labels.append(label)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.training.seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(full_ds)), labels)):
            logger.info("=== Fold %d/%d ===", fold_idx + 1, n_folds)
            train_subset = torch.utils.data.Subset(full_ds, train_idx)
            val_subset = torch.utils.data.Subset(full_ds, val_idx)
            train_loader = DataLoader(
                train_subset,
                batch_size=cfg.data.batch_size,
                shuffle=True,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
                prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=cfg.data.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
                prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
            )
            model = create_model(cfg.training.model, num_classes=NUM_CLASSES_SEG, num_cls_labels=NUM_CLASSES_CLS)
            fold_cfg = Config(
                data=cfg.data,
                optim=cfg.optim,
                training=TrainingConfig(
                    task=cfg.training.task,
                    model=cfg.training.model,
                    epochs=cfg.training.epochs,
                    use_amp=cfg.training.use_amp,
                    use_compile=cfg.training.use_compile,
                    gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                    grad_clip_max_norm=cfg.training.grad_clip_max_norm,
                    seed=cfg.training.seed,
                    log_interval=cfg.training.log_interval,
                    n_folds=n_folds,
                    fold=fold_idx,
                    use_wandb=cfg.training.use_wandb,
                    wandb_project=cfg.training.wandb_project,
                    wandb_run_name=f"{cfg.training.wandb_run_name}_fold{fold_idx}" if cfg.training.wandb_run_name else "",
                ),
                paths=cfg.paths,
            )
            trainer = Trainer(model, train_loader, val_loader, fold_cfg)
            trainer.fit()
        logger.info("K-fold cross-validation complete (%d folds).", n_folds)
    else:
        # ── Standard train/test split ──────────────────────────────
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
        model = create_model(
            cfg.training.model,
            num_classes=NUM_CLASSES_SEG,
            num_cls_labels=NUM_CLASSES_CLS,
        )
        trainer = Trainer(model, train_loader, val_loader, cfg)
        trainer.fit()
    logger.info("Training complete.")
if __name__ == "__main__":
    main()
