"""Evaluation metrics for segmentation and classification.
Provides:
- Dice coefficient, IoU (segmentation)
- Accuracy, Precision, Recall, F1, AUC-ROC, Sensitivity, Specificity (classification)
- Per-class breakdowns
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from src.utils.logging import get_logger
logger = get_logger(__name__)
# ══════════════════════════════════════════════════════════════════════════
# Segmentation metrics
# ══════════════════════════════════════════════════════════════════════════
def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> float:
    """Compute Dice coefficient (returns Python float)."""
    if num_classes > 1:
        pred_softmax = F.softmax(pred, dim=1)
        pred_labels = torch.argmax(pred_softmax, dim=1)
        dice_scores: list[float] = []
        for class_idx in range(1, num_classes):
            pred_mask = (pred_labels == class_idx).float()
            target_mask = (target == class_idx).float()
            inter = torch.sum(pred_mask * target_mask)
            union = torch.sum(pred_mask) + torch.sum(target_mask)
            dice_val = (2.0 * inter + eps) / (union + eps)
            dice_scores.append(dice_val.item())
        return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    else:
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        target_binary = target.float()
        inter = torch.sum(pred_binary * target_binary)
        union = torch.sum(pred_binary) + torch.sum(target_binary)
        return float(((2.0 * inter + eps) / (union + eps)).item())
def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> float:
    """Compute Intersection-over-Union (Jaccard index, returns Python float)."""
    if num_classes > 1:
        pred_softmax = F.softmax(pred, dim=1)
        pred_labels = torch.argmax(pred_softmax, dim=1)
        iou_scores: list[float] = []
        for class_idx in range(1, num_classes):
            pred_mask = (pred_labels == class_idx).float()
            target_mask = (target == class_idx).float()
            inter = torch.sum(pred_mask * target_mask)
            union = torch.sum(pred_mask) + torch.sum(target_mask) - inter
            iou_scores.append(((inter + eps) / (union + eps)).item())
        return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    else:
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        target_binary = target.float()
        inter = torch.sum(pred_binary * target_binary)
        union = torch.sum(pred_binary) + torch.sum(target_binary) - inter
        return float(((inter + eps) / (union + eps)).item())
# ══════════════════════════════════════════════════════════════════════════
# Classification metrics
# ══════════════════════════════════════════════════════════════════════════
def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy()
def classification_metrics(
    pred_logits_or_labels: torch.Tensor,
    target: torch.Tensor,
    average: str = "macro",
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1 (returns Python floats)."""
    if pred_logits_or_labels.ndim > 1:
        preds = torch.argmax(pred_logits_or_labels, dim=1).cpu().numpy()
    else:
        preds = pred_logits_or_labels.cpu().numpy()
    target_np = target.cpu().numpy()
    return {
        "acc": float(accuracy_score(target_np, preds)),
        "precision": float(precision_score(target_np, preds, average=average, zero_division=0)),
        "recall": float(recall_score(target_np, preds, average=average, zero_division=0)),
        "f1": float(f1_score(target_np, preds, average=average, zero_division=0)),
    }
def per_class_metrics(
    pred_logits_or_labels: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4,
) -> dict[str, list[float]]:
    """Compute precision, recall, F1 per class."""
    if pred_logits_or_labels.ndim > 1:
        preds = torch.argmax(pred_logits_or_labels, dim=1).cpu().numpy()
    else:
        preds = pred_logits_or_labels.cpu().numpy()
    target_np = target.cpu().numpy()
    all_labels = list(range(num_classes))
    return {
        "precision_per_class": list(precision_score(target_np, preds, labels=all_labels, average=None, zero_division=0)),
        "recall_per_class": list(recall_score(target_np, preds, labels=all_labels, average=None, zero_division=0)),
        "f1_per_class": list(f1_score(target_np, preds, labels=all_labels, average=None, zero_division=0)),
    }
def auc_roc_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> dict[str, object]:
    """Compute AUC-ROC (macro and per-class).
    Returns dict with roc_auc, roc_auc_per_class, fpr, tpr dicts.
    Requires raw logits (not argmaxed labels).
    """
    target_np = _to_numpy(target)
    probs = F.softmax(pred_logits, dim=1).cpu().numpy()
    target_bin = label_binarize(target_np, classes=list(range(num_classes)))
    fpr: dict[int, np.ndarray] = {}
    tpr: dict[int, np.ndarray] = {}
    roc_auc_per_class: list[float] = []
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(target_bin[:, i], probs[:, i])
        roc_auc_per_class.append(float(auc(fpr[i], tpr[i])))
    return {
        "roc_auc": float(np.mean(roc_auc_per_class)),
        "roc_auc_per_class": roc_auc_per_class,
        "fpr": fpr,
        "tpr": tpr,
    }
def sensitivity_specificity(
    pred_logits_or_labels: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> dict[str, list[float]]:
    """Compute sensitivity (recall) and specificity per class."""
    if pred_logits_or_labels.ndim > 1:
        preds = torch.argmax(pred_logits_or_labels, dim=1).cpu().numpy()
    else:
        preds = pred_logits_or_labels.cpu().numpy()
    target_np = target.cpu().numpy()
    sens_list: list[float] = []
    spec_list: list[float] = []
    for c in range(num_classes):
        tp = float(np.sum((preds == c) & (target_np == c)))
        fn = float(np.sum((preds != c) & (target_np == c)))
        tn = float(np.sum((preds != c) & (target_np != c)))
        fp = float(np.sum((preds == c) & (target_np != c)))
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return {
        "sensitivity_per_class": sens_list,
        "specificity_per_class": spec_list,
    }
def compute_all_cls_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> dict[str, object]:
    """Convenience: compute all classification metrics in one call."""
    metrics: dict[str, object] = classification_metrics(pred_logits, target)
    metrics.update(per_class_metrics(pred_logits, target, num_classes))
    metrics.update(sensitivity_specificity(pred_logits, target, num_classes))
    unique_labels = target.unique()
    if len(unique_labels) > 1:
        metrics.update(auc_roc_metrics(pred_logits, target, num_classes))
    else:
        metrics["roc_auc"] = 0.0
        metrics["roc_auc_per_class"] = [0.0] * num_classes
        metrics["fpr"] = {}
        metrics["tpr"] = {}
    return metrics
def print_metrics(task: str, metrics: dict[str, float]) -> None:
    """Pretty-print metrics relevant to *task*."""
    if task == "seg":
        logger.info("  Dice: %.4f  | IoU: %.4f", metrics.get("dice", 0), metrics.get("iou", 0))
    elif task == "cls":
        logger.info(
            "  Acc: %.4f  | Prec: %.4f  | Recall: %.4f  | F1: %.4f  | AUC-ROC: %.4f",
            metrics.get("acc", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1", 0),
            metrics.get("roc_auc", 0),
        )
    elif task == "joint":
        logger.info("[JOINT] Metrics:")
        seg_m = {k: v for k, v in metrics.items() if k in ("dice", "iou")}
        cls_m = {
            k: v for k, v in metrics.items()
            if isinstance(v, int | float) and k not in ("dice", "iou", "loss")
        }
        logger.info("--- Segmentation ---")
        print_metrics("seg", seg_m)
        logger.info("--- Classification ---")
        print_metrics("cls", cls_m)
