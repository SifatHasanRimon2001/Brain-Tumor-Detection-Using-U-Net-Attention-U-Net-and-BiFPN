"""Experiment reporting — CSV logging, confusion-matrix figures, summaries.

Refactored from the original ``report_utils.py`` with improved structure.
"""

from __future__ import annotations

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.use("Agg")  # non-interactive backend


class ReportManager:
    """Logs training / inference metrics to CSV and saves figures.

    Parameters
    ----------
    root_dir : str
        Top-level report directory (e.g. ``"reports"``).
    model_name : str
        One of ``"unet"``, ``"attunet"``, ``"bifpn"``.
    task : str
        One of ``"seg"``, ``"cls"``, ``"joint"``.
    class_labels : list[str] | None
        Optional class names for confusion-matrix plots.
    """

    def __init__(
        self,
        root_dir: str = "reports",
        model_name: str = "unet",
        task: str = "seg",
        class_labels: list[str] | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.model_name = model_name
        self.task = task
        self.class_labels = class_labels

        self.report_dir = os.path.join(root_dir, model_name, task)
        os.makedirs(self.report_dir, exist_ok=True)

        self.csv_path = os.path.join(self.report_dir, "log.csv")
        self.final_csv_path = os.path.join(self.report_dir, "final_results.csv")
        self.summary_csv_path = os.path.join(root_dir, "summary.csv")

        self._init_csv()
        self._init_summary()

    # ── Initialisation ──────────────────────────────────────────────

    def _init_csv(self) -> None:
        if not os.path.isfile(self.csv_path):
            self._write_csv(self.csv_path, self._header())

    def _init_summary(self) -> None:
        if not os.path.isfile(self.summary_csv_path):
            header = [
                "model", "task", "epoch",
                "train_loss",
                "train_acc", "train_precision", "train_recall", "train_f1",
                "train_dice", "train_iou",
                "test_loss",
                "test_acc", "test_precision", "test_recall", "test_f1",
                "test_dice", "test_iou",
            ]
            with open(self.summary_csv_path, mode="w", newline="") as f:
                csv.writer(f).writerow(header)

    @staticmethod
    def _header() -> list[str]:
        return [
            "epoch",
            "train_loss",
            "train_acc", "train_precision", "train_recall",
            "train_f1", "train_dice", "train_iou",
            "test_loss",
            "test_acc", "test_precision", "test_recall",
            "test_f1", "test_dice", "test_iou",
            "confusion_matrix",
        ]

    @staticmethod
    def _write_csv(path: str, row: list) -> None:
        with open(path, mode="a", newline="") as f:
            csv.writer(f).writerow(row)

    # ── Logging ─────────────────────────────────────────────────────

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        train_metrics: dict[str, float],
        test_metrics: dict[str, float],
        confusion_matrix: np.ndarray | None = None,
    ) -> None:
        """Append one row to the per-model CSV and update the summary CSV."""
        cm_str = ""
        if confusion_matrix is not None:
            cm_str = np.array2string(confusion_matrix, separator=", ")
            self.save_confusion_matrix(confusion_matrix, epoch)

        row = [
            epoch,
            train_loss,
            train_metrics.get("acc", ""),
            train_metrics.get("precision", ""),
            train_metrics.get("recall", ""),
            train_metrics.get("f1", ""),
            train_metrics.get("dice", ""),
            train_metrics.get("iou", ""),
            test_loss,
            test_metrics.get("acc", ""),
            test_metrics.get("precision", ""),
            test_metrics.get("recall", ""),
            test_metrics.get("f1", ""),
            test_metrics.get("dice", ""),
            test_metrics.get("iou", ""),
            cm_str,
        ]

        # Append to per-model log
        self._write_csv(self.csv_path, row)

        # Overwrite final_results with the latest row
        self._write_csv(self.final_csv_path, self._header())
        self._write_csv(self.final_csv_path, row)

        # Update summary
        self._update_summary(epoch, row)

    # ── Summary management ──────────────────────────────────────────

    def _update_summary(self, epoch: int, row: list) -> None:
        new_entry = [self.model_name, self.task, row[0], *row[1:-1]]

        rows: list[list[str]] = []
        if os.path.isfile(self.summary_csv_path):
            with open(self.summary_csv_path, mode="r") as f:
                reader = list(csv.reader(f))
                rows = reader[1:]  # skip header

        # Replace existing entry for this (model, task) or append
        rows = [
            r for r in rows
            if not (r[0] == self.model_name and r[1] == self.task)
        ]
        rows.append(list(map(str, new_entry)))

        header = [
            "model", "task", "epoch",
            "train_loss", "train_acc", "train_precision", "train_recall",
            "train_f1", "train_dice", "train_iou",
            "test_loss", "test_acc", "test_precision", "test_recall",
            "test_f1", "test_dice", "test_iou",
        ]
        with open(self.summary_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    # ── Confusion matrix figure ─────────────────────────────────────

    def save_confusion_matrix(
        self,
        cm: np.ndarray,
        epoch: int | None = None,
        labels: list[str] | None = None,
    ) -> None:
        """Save a confusion-matrix PNG to the report directory."""
        labels = labels or self.class_labels
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d", ax=ax)
        ax.set_title(
            f"Confusion Matrix (epoch {epoch})" if epoch else "Confusion Matrix"
        )
        filename = f"confusion_matrix_epoch{epoch}.png" if epoch else "confusion_matrix.png"
        fig.savefig(os.path.join(self.report_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
