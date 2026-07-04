"""Modular training loop for segmentation / classification / joint tasks.

Improvements over the original monolithic ``run_train.py``:

- Avoids concatenating all validation outputs into a single giant tensor
  (which caused OOM for joint tasks). Instead computes metrics per-batch
  and accumulates.
- Proper checkpoint management: saves best model based on task-specific score.
- Gradient accumulation, AMP, logging.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CLASS_MAP, NUM_CLASSES_CLS, Config
from src.models.losses import DiceLoss
from src.models.metrics import (
    classification_metrics,
    dice_coefficient,
    iou_score,
    print_metrics,
)
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import get_logger
from src.utils.plotting import (
    plot_final_metrics_bars,
    plot_loss_curves,
    plot_metrics_radar,
    plot_seg_metric_curves,
    plot_cls_metric_curves,
    plot_training_summary,
)
from src.utils.reporting import ReportManager

logger = get_logger(__name__)


class Trainer:
    """Encapsulates the training and validation logic.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
    val_loader : DataLoader
    cfg : Config
        Top-level configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.model.to(self.device)

        # Losses
        self.criterion_seg = DiceLoss()
        self.criterion_cls = nn.CrossEntropyLoss()

        # Optimiser & scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.optim.learning_rate,
            weight_decay=cfg.optim.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.epochs
        )

        # AMP
        self.scaler = GradScaler(enabled=cfg.training.use_amp)

        # Reporting
        self.reporter = ReportManager(
            root_dir=cfg.paths.report_dir,
            model_name=cfg.training.model,
            task=cfg.training.task,
            class_labels=list(CLASS_MAP.keys()),
        )

        # Best-model tracking
        self.best_score: float = -1.0
        self.best_ckpt_dir = cfg.checkpoint_path
        os.makedirs(self.best_ckpt_dir, exist_ok=True)

        # Epoch history for plotting
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.val_metric_history: dict[str, list[float]] = {}

    # ── Public API ──────────────────────────────────────────────────

    def fit(self) -> None:
        """Run the full training loop."""
        for epoch in range(1, self.cfg.training.epochs + 1):
            current_lr = self.scheduler.get_last_lr()[0]
            logger.info(
                "Epoch %d/%d  |  LR=%.6f",
                epoch,
                self.cfg.training.epochs,
                current_lr,
            )

            train_loss = self._train_one_epoch()
            val_results, cm = self._validate()

            logger.info(
                "Train loss: %.4f  |  Val loss: %.4f",
                train_loss,
                val_results["loss"],
            )
            print_metrics(self.cfg.training.task, val_results)

            # Log
            self.reporter.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                test_loss=val_results["loss"],
                train_metrics={},
                test_metrics=val_results,
                confusion_matrix=cm,
            )

            # Store histories
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_results["loss"])
            for k, v in val_results.items():
                if isinstance(v, (int, float)):
                    self.val_metric_history.setdefault(k, []).append(v)

            # Checkpoint
            score = self._compute_score(val_results)
            if score > self.best_score:
                self.best_score = score
                save_checkpoint(
                    self.model.state_dict(),
                    self.best_ckpt_dir,
                    "best.ckpt",
                )
                logger.info("Saved best model (score=%.4f)", score)

            self.scheduler.step()

        # ── Generate plots after training completes ─────────────────
        self._generate_plots()

    def _generate_plots(self) -> None:
        """Generate all training summary plots and save to report directory."""
        report_dir = self.reporter.report_dir

        # 1. Loss curves
        plot_loss_curves(
            self.train_loss_history,
            self.val_loss_history,
            os.path.join(report_dir, "loss_curves.png"),
        )

        # 2. Task-specific metric curves
        task = self.cfg.training.task
        if task in ("seg", "joint"):
            plot_seg_metric_curves(
                self.val_metric_history,
                os.path.join(report_dir, "seg_metric_curves.png"),
            )
        if task in ("cls", "joint"):
            plot_cls_metric_curves(
                self.val_metric_history,
                os.path.join(report_dir, "cls_metric_curves.png"),
            )

        # 3. Training summary (multi-panel figure)
        plot_training_summary(
            self.train_loss_history,
            self.val_loss_history,
            self.val_metric_history,
            os.path.join(report_dir, "training_summary.png"),
            task=task,
        )

        # 4. Final metrics bar chart
        if self.val_metric_history:
            final_metrics = {
                k: v[-1] for k, v in self.val_metric_history.items()
                if v
            }
            plot_final_metrics_bars(
                final_metrics,
                os.path.join(report_dir, "final_metrics_bars.png"),
            )
            plot_metrics_radar(
                final_metrics,
                os.path.join(report_dir, "metrics_radar.png"),
            )

        logger.info("Training plots saved to %s", report_dir)

    # ── Training step ───────────────────────────────────────────────

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        accum_steps = self.cfg.training.gradient_accumulation_steps

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            loss = self._compute_train_loss(batch) / accum_steps
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.training.grad_clip_max_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accum_steps

        return total_loss / len(self.train_loader)

    def _compute_train_loss(self, batch) -> torch.Tensor:
        task = self.cfg.training.task

        with autocast(enabled=self.cfg.training.use_amp):
            if task == "seg":
                imgs, masks = batch
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                seg_out, _ = self.model(imgs)
                return self.criterion_seg(seg_out, masks)

            elif task == "cls":
                imgs, labels = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                _, cls_out = self.model(imgs)
                return self.criterion_cls(cls_out, labels)

            else:  # joint
                imgs, masks, labels = batch
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                seg_out, cls_out = self.model(imgs)
                loss_seg = self.criterion_seg(seg_out, masks)
                loss_cls = self.criterion_cls(cls_out, labels)
                return loss_seg + loss_cls

    # ── Validation step (memory-efficient) ─────────────────────────

    def _validate(self) -> tuple[dict, object]:
        """Validate and return ``(metrics_dict, confusion_matrix_or_None)``.

        **Memory fix**: Instead of storing all outputs and concatenating
        at the end, we accumulate batch-level metrics to avoid OOM.
        """
        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        task = self.cfg.training.task
        num_classes = 1  # binary seg

        # Accumulators for per-batch metrics
        dice_list: list[float] = []
        iou_list: list[float] = []
        cls_preds_list: list[int] = []
        cls_labels_list: list[int] = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                with autocast(enabled=self.cfg.training.use_amp):
                    if task == "seg":
                        imgs, masks = batch
                        imgs, masks = imgs.to(self.device), masks.to(self.device)
                        seg_out, _ = self.model(imgs)
                        loss = self.criterion_seg(seg_out, masks)
                        dice_list.append(dice_coefficient(seg_out, masks, num_classes))
                        iou_list.append(iou_score(seg_out, masks, num_classes))

                    elif task == "cls":
                        imgs, labels = batch
                        imgs, labels = imgs.to(self.device), labels.to(self.device)
                        _, cls_out = self.model(imgs)
                        loss = self.criterion_cls(cls_out, labels)
                        preds = torch.argmax(cls_out, dim=1)
                        cls_preds_list.extend(preds.cpu().tolist())
                        cls_labels_list.extend(labels.cpu().tolist())

                    else:  # joint
                        imgs, masks, labels = batch
                        imgs = imgs.to(self.device)
                        masks = masks.to(self.device)
                        labels = labels.to(self.device)
                        seg_out, cls_out = self.model(imgs)
                        loss_seg = self.criterion_seg(seg_out, masks)
                        loss_cls = self.criterion_cls(cls_out, labels)
                        loss = loss_seg + loss_cls
                        dice_list.append(dice_coefficient(seg_out, masks, num_classes))
                        iou_list.append(iou_score(seg_out, masks, num_classes))
                        preds = torch.argmax(cls_out, dim=1)
                        cls_preds_list.extend(preds.cpu().tolist())
                        cls_labels_list.extend(labels.cpu().tolist())

                total_loss += loss.item()
                n_batches += 1

        results: dict[str, float] = {"loss": total_loss / max(n_batches, 1)}

        if task in ("seg", "joint"):
            results["dice"] = sum(dice_list) / len(dice_list) if dice_list else 0.0
            results["iou"] = sum(iou_list) / len(iou_list) if iou_list else 0.0

        cm = None
        if task in ("cls", "joint") and cls_preds_list:
            cls_metrics = classification_metrics(
                torch.tensor(cls_preds_list),
                torch.tensor(cls_labels_list),
            )
            results.update(cls_metrics)
            cm = confusion_matrix(cls_labels_list, cls_preds_list, labels=list(range(NUM_CLASSES_CLS)))

        return results, cm

    # ── Scoring ─────────────────────────────────────────────────────

    def _compute_score(self, results: dict[str, float]) -> float:
        task = self.cfg.training.task
        if task == "seg":
            return results.get("dice", 0.0)
        elif task == "cls":
            return results.get("f1", 0.0)
        else:  # joint
            return results.get("dice", 0.0) + results.get("f1", 0.0)
