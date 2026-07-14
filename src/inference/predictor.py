"""Inference / evaluation logic for trained models.
**Fix**: Dice is now properly computed for joint tasks during inference
(previously only IoU was computed, leaving dice = 0 in the report).
"""
from __future__ import annotations
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import CLASS_MAP, NUM_CLASSES_CLS, NUM_CLASSES_SEG, Config
from src.data.dataset import BriscDataset
from src.models import create_model
from src.models.metrics import (
    classification_metrics,
    compute_all_cls_metrics,
    dice_coefficient,
    iou_score,
)
from src.utils.checkpoint import load_checkpoint
from src.utils.logging import get_logger
from src.utils.plotting import (
    plot_final_metrics_bars,
    plot_metrics_radar,
    plot_per_class_bars,
    plot_roc_curves,
    plot_sensitivity_specificity_bars,
)
from src.utils.reporting import ReportManager
logger = get_logger(__name__)
class Predictor:
    """Run inference with a trained model and produce evaluation reports.
    Parameters
    ----------
    cfg : Config
        Configuration (task, model name, paths, …).
    ckpt_path : str | None
        Optional explicit checkpoint path. If ``None``, uses ``cfg.best_ckpt_path``.
    """
    def __init__(self, cfg: Config, ckpt_path: str | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # Build model & load checkpoint
        ckpt = ckpt_path or cfg.best_ckpt_path
        self.model = create_model(
            cfg.training.model,
            num_classes=NUM_CLASSES_SEG,
            num_cls_labels=NUM_CLASSES_CLS,
        ).to(self.device)
        load_checkpoint(self.model, ckpt, device=str(self.device))
        self.model.eval()
        logger.info("Loaded checkpoint from %s", ckpt)
        # Reporter
        self.reporter = ReportManager(
            root_dir=cfg.paths.report_dir,
            model_name=cfg.training.model,
            task=cfg.training.task,
        )
        # Output directory for saved predictions
        self.out_dir = os.path.join(
            cfg.paths.output_dir, cfg.training.model, cfg.training.task
        )
        os.makedirs(self.out_dir, exist_ok=True)
    # ── Public API ──────────────────────────────────────────────────
    def run(self) -> dict[str, float]:
        """Run inference and return final metrics."""
        ds = BriscDataset(
            self.cfg.data.data_root,
            split="test",
            task=self.cfg.training.task,
            img_size=self.cfg.data.img_size,
        )
        loader = DataLoader(
            ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers and self.cfg.data.num_workers > 0,
            prefetch_factor=self.cfg.data.prefetch_factor if self.cfg.data.num_workers > 0 else None,
        )
        total_iou: list[float] = []
        total_dice: list[float] = []
        cls_preds_list: list[int] = []
        cls_labels_list: list[int] = []
        cls_logits_all: list[torch.Tensor] = []  # for ROC curves
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader, desc="Inferring")):
                task = self.cfg.training.task
                if task == "seg":
                    imgs, masks = batch
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    seg_out, _ = self.model(imgs)
                    total_iou.append(iou_score(seg_out, masks, NUM_CLASSES_SEG))
                    total_dice.append(dice_coefficient(seg_out, masks, NUM_CLASSES_SEG))
                    for j in range(imgs.shape[0]):
                        self._save_mask(seg_out, i * imgs.shape[0] + j, batch_idx=j)
                elif task == "cls":
                    imgs, labels = batch
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    _, cls_out = self.model(imgs)
                    preds = torch.argmax(cls_out, dim=1)
                    cls_preds_list.extend(preds.cpu().tolist())
                    cls_labels_list.extend(labels.cpu().tolist())
                    cls_logits_all.append(cls_out)
                else:  # joint
                    imgs, masks, labels = batch
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)
                    seg_out, cls_out = self.model(imgs)
                    # Segmentation metrics
                    total_iou.append(iou_score(seg_out, masks, NUM_CLASSES_SEG))
                    total_dice.append(dice_coefficient(seg_out, masks, NUM_CLASSES_SEG))
                    # Classification metrics
                    preds = torch.argmax(cls_out, dim=1)
                    cls_preds_list.extend(preds.cpu().tolist())
                    cls_labels_list.extend(labels.cpu().tolist())
                    cls_logits_all.append(cls_out)
                    for j in range(imgs.shape[0]):
                        self._save_mask(seg_out, i * imgs.shape[0] + j, batch_idx=j)
        # Build final metrics dict
        test_metrics: dict[str, float] = {
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "dice": 0.0,
            "iou": 0.0,
        }
        if total_iou:
            test_metrics["iou"] = float(np.mean(total_iou))
            logger.info("Mean IoU = %.4f", test_metrics["iou"])
        if total_dice:
            test_metrics["dice"] = float(np.mean(total_dice))
            logger.info("Mean Dice = %.4f", test_metrics["dice"])
        if cls_preds_list:
            cls_metrics = classification_metrics(
                torch.tensor(cls_preds_list),
                torch.tensor(cls_labels_list),
            )
            test_metrics.update(cls_metrics)
            logger.info("Classification metrics: %s", cls_metrics)
        # ── Generate evaluation plots ──────────────────────────────
        self._generate_eval_plots(
            test_metrics, cls_logits_all, cls_labels_list, total_dice, total_iou,
        )
        # Write report (epoch=0 signals final inference)
        try:
            self.reporter.log_epoch(
                epoch=0,
                train_loss=0.0,
                test_loss=0.0,
                train_metrics={},
                test_metrics=test_metrics,
            )
        except (PermissionError, OSError) as e:
            logger.warning("Could not write report (file locked or inaccessible): %s", e)
        return test_metrics
    # ── Evaluation plots ────────────────────────────────────────
    def _generate_eval_plots(
        self,
        test_metrics: dict,
        cls_logits_all: list[torch.Tensor],
        cls_labels_list: list[int],
        total_dice: list[float],
        total_iou: list[float],
    ) -> None:
        """Generate inference-time evaluation plots."""
        report_dir = self.reporter.report_dir
        task = self.cfg.training.task
        # 1. Final metrics bar chart
        plot_final_metrics_bars(
            test_metrics,
            os.path.join(report_dir, "final_metrics_bars.png"),
            title=f"Inference Metrics — {task.upper()}",
        )
        plot_metrics_radar(
            test_metrics,
            os.path.join(report_dir, "metrics_radar.png"),
        )
        # 2. ROC curves (classification-only or joint)
        if task in ("cls", "joint") and cls_logits_all:
            logits_cat = torch.cat(cls_logits_all, dim=0)
            labels_t = torch.tensor(cls_labels_list)
            all_metrics = compute_all_cls_metrics(logits_cat, labels_t, NUM_CLASSES_CLS)
            plot_roc_curves(
                all_metrics["fpr"],
                all_metrics["tpr"],
                all_metrics["roc_auc_per_class"],
                os.path.join(report_dir, "roc_curves.png"),
                class_labels=list(CLASS_MAP.keys()),
            )
            # 3. Per-class metrics
            plot_per_class_bars(
                all_metrics.get("precision_per_class"),
                all_metrics.get("recall_per_class"),
                all_metrics.get("f1_per_class"),
                os.path.join(report_dir, "per_class_metrics.png"),
                class_labels=list(CLASS_MAP.keys()),
            )
            plot_sensitivity_specificity_bars(
                all_metrics.get("sensitivity_per_class", []),
                all_metrics.get("specificity_per_class", []),
                os.path.join(report_dir, "sens_spec_per_class.png"),
                class_labels=list(CLASS_MAP.keys()),
            )
        # 4. Segmentation metrics (if applicable)
        if task in ("seg", "joint") and total_dice:
            plot_final_metrics_bars(
                {"dice": float(np.mean(total_dice)), "iou": float(np.mean(total_iou))},
                os.path.join(report_dir, "seg_metrics_bars.png"),
                title="Segmentation Metrics",
            )
        logger.info("Evaluation plots saved to %s", report_dir)
    # ── Internal helper ─────────────────────────────────────────────
    def _save_mask(self, seg_out: torch.Tensor, idx: int, batch_idx: int = 0) -> None:
        pred_mask = (torch.sigmoid(seg_out) > 0.5).cpu().numpy()[batch_idx, 0]
        cv2.imwrite(
            os.path.join(self.out_dir, f"pred_mask_{idx}.png"),
            (pred_mask * 255).astype(np.uint8),
        )
