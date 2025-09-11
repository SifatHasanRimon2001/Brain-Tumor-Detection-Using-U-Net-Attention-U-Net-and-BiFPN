import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
class ReportManager:
    def __init__(self, root_dir="report", model_name="unet", task="seg"):
        self.root_dir = root_dir
        self.model_name = model_name
        self.task = task
        self.report_dir = os.path.join(root_dir, model_name, task)
        os.makedirs(self.report_dir, exist_ok=True)
        self.csv_path = os.path.join(self.report_dir, "log.csv")
        self.final_csv_path = os.path.join(self.report_dir, "final_results.csv")
        self.summary_csv_path = os.path.join(root_dir, "summary.csv")
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss", "train_acc", "train_precision", "train_recall",
                    "train_f1", "train_dice", "train_iou",
                    "test_loss", "test_acc", "test_precision", "test_recall",
                    "test_f1", "test_dice", "test_iou",
                    "confusion_matrix"
                ])
        if not os.path.isfile(self.summary_csv_path):
            with open(self.summary_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "model", "task", "epoch",
                    "train_loss", "train_acc", "train_precision", "train_recall",
                    "train_f1", "train_dice", "train_iou",
                    "test_loss", "test_acc", "test_precision", "test_recall",
                    "test_f1", "test_dice", "test_iou"
                ])
    def log_epoch(self, epoch, train_loss, test_loss, train_metrics, test_metrics, confusion_matrix=None):
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
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        with open(self.final_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_acc", "train_precision", "train_recall", "train_f1", "train_dice", "train_iou",
                "test_loss", "test_acc", "test_precision", "test_recall", "test_f1", "test_dice", "test_iou",
                "confusion_matrix"
            ])
            writer.writerow(row)
        self._update_summary(epoch, row)
    def _update_summary(self, epoch, row):
        new_entry = [
            self.model_name, self.task, row[0],
            *row[1:-1],
        ]
        rows = []
        if os.path.isfile(self.summary_csv_path):
            with open(self.summary_csv_path, mode="r") as f:
                reader = list(csv.reader(f))
                header, rows = reader[0], reader[1:]
        rows = [r for r in rows if not (r[0] == self.model_name and r[1] == self.task)]
        rows.append(list(map(str, new_entry)))
        with open(self.summary_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "task", "epoch",
                "train_loss", "train_acc", "train_precision", "train_recall",
                "train_f1", "train_dice", "train_iou",
                "test_loss", "test_acc", "test_precision", "test_recall",
                "test_f1", "test_dice", "test_iou"
            ])
            writer.writerows(rows)
    def save_confusion_matrix(self, cm, epoch=None, labels=None):
        plt.figure(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
        plt.title(f"Confusion Matrix (epoch {epoch})" if epoch else "Confusion Matrix")
        filename = f"confusion_matrix_epoch{epoch}.png" if epoch else "confusion_matrix.png"
        plt.savefig(os.path.join(self.report_dir, filename))
        plt.close()