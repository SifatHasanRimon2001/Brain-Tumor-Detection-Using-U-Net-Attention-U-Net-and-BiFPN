"""Comprehensive graph generation for all metrics.

Generates and saves plots for:
1. Training & validation loss curves
2. Metric curves over epochs (acc, precision, recall, f1, dice, iou)
3. ROC curves with AUC for classification
4. Per-class precision/recall/F1 bar charts
5. Sensitivity/Specificity per class
6. Combined metrics overview figure (radar or bar)
7. Final inference metrics comparison

All figures are saved to the report directory for each model/task.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ── Style ─────────────────────────────────────────────────────────────────

LABELS_CLS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. Loss curves
# ════════════════════════════════════════════════════════════════════════════


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str,
    title: str = "Training & Validation Loss",
) -> None:
    """Plot train/val loss over epochs."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "o-", color=COLORS[0], label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, "s--", color=COLORS[1], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 2. Metric curves over epochs
# ════════════════════════════════════════════════════════════════════════════


def plot_metric_curves(
    history: dict[str, list[float]],
    save_path: str,
    metrics_to_plot: list[str] | None = None,
    title: str = "Validation Metrics",
) -> None:
    """Plot one or more validation metrics over epochs."""
    _ensure_dir(save_path)
    if not history or not any(history.values()):
        return  # nothing to plot
    fig, ax = plt.subplots(figsize=(10, 6))

    keys = metrics_to_plot or [k for k in history if k != "loss"]
    first_val = next((v for v in history.values() if v), [])
    epochs = range(1, len(first_val) + 1) if first_val else []

    for i, key in enumerate(keys):
        values = history.get(key, [])
        if values:
            ax.plot(epochs, values, "o-", color=COLORS[i % len(COLORS)],
                    label=key.replace("_", " ").title(), linewidth=2)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_seg_metric_curves(
    history: dict[str, list[float]],
    save_path: str,
) -> None:
    """Plot Dice and IoU curves for segmentation."""
    plot_metric_curves(
        history, save_path,
        metrics_to_plot=["dice", "iou"],
        title="Segmentation Metrics (Dice & IoU)",
    )


def plot_cls_metric_curves(
    history: dict[str, list[float]],
    save_path: str,
) -> None:
    """Plot accuracy, precision, recall, F1 curves for classification."""
    plot_metric_curves(
        history, save_path,
        metrics_to_plot=["acc", "precision", "recall", "f1"],
        title="Classification Metrics (Acc, Prec, Recall, F1)",
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. ROC curves
# ════════════════════════════════════════════════════════════════════════════


def plot_roc_curves(
    fpr: dict[int, np.ndarray],
    tpr: dict[int, np.ndarray],
    roc_auc_per_class: list[float],
    save_path: str,
    class_labels: list[str] | None = None,
) -> None:
    """Plot ROC curves for each class with AUC values."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 7))
    labels = class_labels or [f"Class {i}" for i in range(len(roc_auc_per_class))]

    for i, (cls_label, auc_val) in enumerate(zip(labels, roc_auc_per_class)):
        ax.plot(
            fpr[i], tpr[i], linewidth=2,
            label=f"{cls_label} (AUC = {auc_val:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 4. Per-class bar charts
# ════════════════════════════════════════════════════════════════════════════


def plot_per_class_bars(
    precision_per_class: list[float] | None,
    recall_per_class: list[float] | None,
    f1_per_class: list[float] | None,
    save_path: str,
    class_labels: list[str] | None = None,
) -> None:
    """Grouped bar chart of precision, recall, F1 per class."""
    _ensure_dir(save_path)
    labels = class_labels or [f"C{i}" for i in range(
        max(len(precision_per_class or []),
            len(recall_per_class or []),
            len(f1_per_class or []))
    )]
    n = len(labels)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    if precision_per_class:
        ax.bar(x - width, precision_per_class, width, label="Precision", alpha=0.85)
    if recall_per_class:
        ax.bar(x, recall_per_class, width, label="Recall", alpha=0.85)
    if f1_per_class:
        ax.bar(x + width, f1_per_class, width, label="F1-Score", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Per-Class Metrics", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_specificity_bars(
    sensitivity_per_class: list[float],
    specificity_per_class: list[float],
    save_path: str,
    class_labels: list[str] | None = None,
) -> None:
    """Bar chart of sensitivity and specificity per class."""
    _ensure_dir(save_path)
    labels = class_labels or [f"C{i}" for i in range(len(sensitivity_per_class))]
    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, sensitivity_per_class, width, label="Sensitivity", alpha=0.85)
    ax.bar(x + width / 2, specificity_per_class, width, label="Specificity", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Sensitivity & Specificity Per Class", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 5. Metrics overview (radar chart)
# ════════════════════════════════════════════════════════════════════════════


def plot_metrics_radar(
    metrics: dict[str, float],
    save_path: str,
    title: str = "Metrics Overview",
) -> None:
    """Radar chart showing a snapshot of all metrics."""
    _ensure_dir(save_path)
    keys = [k for k in metrics if isinstance(metrics[k], (int, float))
            and k != "loss"]
    values = [metrics[k] for k in keys]

    if not keys:
        return

    n = len(keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.fill(angles, values, alpha=0.25)
    ax.plot(angles, values, linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.replace("_", " ").title() for k in keys], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 6. Combined final metrics bar comparison
# ════════════════════════════════════════════════════════════════════════════


def plot_final_metrics_bars(
    metrics: dict[str, float],
    save_path: str,
    title: str = "Final Evaluation Metrics",
) -> None:
    """Horizontal bar chart of final evaluation metrics."""
    _ensure_dir(save_path)
    keys = [k for k in metrics if isinstance(metrics[k], (int, float))
            and k not in ("loss", "epoch")]
    values = [metrics[k] for k in keys]

    if not keys:
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(keys) * 0.5)))
    bars = ax.barh(keys, values, color=COLORS[:len(keys)], alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 7. Master training plot (all-in-one figure)
# ════════════════════════════════════════════════════════════════════════════


def plot_training_summary(
    train_losses: list[float],
    val_losses: list[float],
    val_history: dict[str, list[float]],
    save_path: str,
    task: str = "seg",
) -> None:
    """Generate a multi-panel summary figure for a full training run."""
    _ensure_dir(save_path)

    # Determine sub-plot layout
    if task == "joint":
        n_cols = 3
        fig, axes = plt.subplots(2, n_cols, figsize=(18, 10))
        axes = axes.flatten()
    else:
        n_cols = 2
        fig, axes = plt.subplots(1 if task == "seg" else 2, n_cols,
                                 figsize=(14, 5), squeeze=False)
        axes = axes.flatten()

    idx = 0
    epochs = range(1, len(train_losses) + 1)

    # Panel 1: Loss
    axes[idx].plot(epochs, train_losses, "o-", label="Train Loss", linewidth=2)
    axes[idx].plot(epochs, val_losses, "s--", label="Val Loss", linewidth=2)
    axes[idx].set_xlabel("Epoch")
    axes[idx].set_ylabel("Loss")
    axes[idx].set_title("Loss Curves", fontweight="bold")
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    # Panel 2: Segmentation metrics (if applicable)
    if task in ("seg", "joint"):
        axes[idx].plot(epochs, val_history.get("dice", []), "o-", label="Dice", linewidth=2)
        axes[idx].plot(epochs, val_history.get("iou", []), "s--", label="IoU", linewidth=2)
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Score")
        axes[idx].set_title("Segmentation Metrics", fontweight="bold")
        axes[idx].legend(fontsize=9)
        axes[idx].set_ylim(0, 1.05)
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # Panel 3: Classification metrics (if applicable)
    if task in ("cls", "joint"):
        axes[idx].plot(epochs, val_history.get("acc", []), "o-", label="Acc", linewidth=2)
        axes[idx].plot(epochs, val_history.get("precision", []), "s--", label="Prec", linewidth=2)
        axes[idx].plot(epochs, val_history.get("recall", []), "D-.", label="Recall", linewidth=2)
        axes[idx].plot(epochs, val_history.get("f1", []), "x-", label="F1", linewidth=2)
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Score")
        axes[idx].set_title("Classification Metrics", fontweight="bold")
        axes[idx].legend(fontsize=9)
        axes[idx].set_ylim(0, 1.05)
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # Panel 4: ROC-AUC (if available)
    if val_history.get("roc_auc", []):
        axes[idx].plot(epochs, val_history["roc_auc"], "o-", color="purple",
                       label="AUC-ROC", linewidth=2)
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("AUC-ROC")
        axes[idx].set_title("ROC-AUC", fontweight="bold")
        axes[idx].legend(fontsize=9)
        axes[idx].set_ylim(0, 1.05)
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # Hide unused subplots
    for j in range(idx, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Training Summary — {task.upper()}", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
