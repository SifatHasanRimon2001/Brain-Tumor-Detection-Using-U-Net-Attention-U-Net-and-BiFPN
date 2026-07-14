"""Tests for evaluation metrics."""
from __future__ import annotations
import torch
from src.models.metrics import (
    classification_metrics,
    compute_all_cls_metrics,
    dice_coefficient,
    iou_score,
    per_class_metrics,
    sensitivity_specificity,
)
class TestDiceCoefficient:
    """Tests for segmentation Dice coefficient."""
    def test_perfect_match(self) -> None:
        pred = torch.ones(1, 1, 32, 32) * 5.0
        target = torch.ones(1, 1, 32, 32)
        dice = dice_coefficient(pred, target, num_classes=1)
        assert dice > 0.99
    def test_no_overlap(self) -> None:
        pred = torch.ones(1, 1, 32, 32) * 5.0
        target = torch.zeros(1, 1, 32, 32)
        dice = dice_coefficient(pred, target, num_classes=1)
        assert dice < 0.01
    def test_returns_float(self) -> None:
        pred = torch.randn(1, 1, 16, 16)
        target = torch.zeros(1, 1, 16, 16)
        result = dice_coefficient(pred, target, num_classes=1)
        assert isinstance(result, float)
class TestIoUScore:
    """Tests for Intersection-over-Union."""
    def test_perfect_match(self) -> None:
        pred = torch.ones(1, 1, 32, 32) * 5.0
        target = torch.ones(1, 1, 32, 32)
        iou = iou_score(pred, target, num_classes=1)
        assert iou > 0.99
    def test_no_overlap(self) -> None:
        pred = torch.ones(1, 1, 32, 32) * 5.0
        target = torch.zeros(1, 1, 32, 32)
        iou = iou_score(pred, target, num_classes=1)
        assert iou < 0.01
    def test_returns_float(self) -> None:
        pred = torch.randn(1, 1, 16, 16)
        target = torch.zeros(1, 1, 16, 16)
        result = iou_score(pred, target, num_classes=1)
        assert isinstance(result, float)
class TestClassificationMetrics:
    """Tests for classification metric functions."""
    def test_perfect_predictions(self) -> None:
        preds = torch.tensor([0, 1, 2, 3, 0, 1])
        target = torch.tensor([0, 1, 2, 3, 0, 1])
        metrics = classification_metrics(preds, target)
        assert metrics["acc"] == 1.0
        assert metrics["f1"] == 1.0
    def test_random_predictions(self) -> None:
        preds = torch.tensor([0, 1, 0, 1])
        target = torch.tensor([0, 0, 1, 1])
        metrics = classification_metrics(preds, target)
        assert 0.0 <= metrics["acc"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
    def test_logit_input(self) -> None:
        logits = torch.tensor([
            [10.0, -10.0, -10.0, -10.0],
            [-10.0, 10.0, -10.0, -10.0],
        ])
        target = torch.tensor([0, 1])
        metrics = classification_metrics(logits, target)
        assert metrics["acc"] == 1.0
    def test_per_class_metrics_shape(self) -> None:
        preds = torch.tensor([0, 1, 2, 3])
        target = torch.tensor([0, 1, 2, 3])
        result = per_class_metrics(preds, target, num_classes=4)
        assert len(result["precision_per_class"]) == 4
        assert len(result["recall_per_class"]) == 4
        assert len(result["f1_per_class"]) == 4
    def test_sensitivity_specificity(self) -> None:
        preds = torch.tensor([0, 0, 1, 1])
        target = torch.tensor([0, 1, 0, 1])
        result = sensitivity_specificity(preds, target, num_classes=2)
        assert len(result["sensitivity_per_class"]) == 2
        assert len(result["specificity_per_class"]) == 2
    def test_compute_all_cls_metrics(self) -> None:
        logits = torch.randn(20, 4)
        target = torch.randint(0, 4, (20,))
        result = compute_all_cls_metrics(logits, target, num_classes=4)
        assert "acc" in result
        assert "f1" in result
        assert "roc_auc" in result
