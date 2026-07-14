"""Tests for loss functions."""
from __future__ import annotations
import torch
from src.models.losses import CombinedJointLoss, DiceLoss
class TestDiceLoss:
    """Tests for soft Dice loss."""
    def test_perfect_prediction_loss_near_zero(self) -> None:
        pred = torch.randn(2, 1, 32, 32)
        target = (torch.sigmoid(pred) > 0.5).float()
        loss = DiceLoss()(pred, target)
        assert loss.item() < 0.1, f"Loss too high for perfect prediction: {loss.item()}"
    def test_opposite_prediction_loss_near_one(self) -> None:
        pred = torch.ones(2, 1, 32, 32) * 5.0
        target = torch.zeros(2, 1, 32, 32)
        loss = DiceLoss()(pred, target)
        assert loss.item() > 0.5, f"Loss too low for opposite prediction: {loss.item()}"
    def test_loss_is_differentiable(self) -> None:
        pred = torch.randn(1, 1, 16, 16, requires_grad=True)
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 4:12, 4:12] = 1.0
        loss = DiceLoss()(pred, target)
        loss.backward()
        assert pred.grad is not None
    def test_multiclass(self) -> None:
        pred = torch.randn(2, 3, 16, 16)
        target = torch.randint(0, 3, (2, 16, 16))
        loss = DiceLoss()(pred, target)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0.0
class TestCombinedJointLoss:
    """Tests for combined Dice + CrossEntropy loss."""
    def test_returns_three_values(self) -> None:
        seg_out = torch.randn(2, 1, 16, 16)
        cls_out = torch.randn(2, 4)
        seg_target = torch.zeros(2, 1, 16, 16)
        cls_target = torch.tensor([0, 2])
        total, seg_loss, cls_loss = CombinedJointLoss()(seg_out, cls_out, seg_target, cls_target)
        assert total.ndim == 0
        assert seg_loss.ndim == 0
        assert cls_loss.ndim == 0
    def test_weighted_combination(self) -> None:
        seg_out = torch.randn(1, 1, 16, 16)
        cls_out = torch.randn(1, 4)
        seg_target = torch.zeros(1, 1, 16, 16)
        cls_target = torch.tensor([1])
        total, seg_loss, cls_loss = CombinedJointLoss(seg_weight=2.0, cls_weight=0.5)(
            seg_out, cls_out, seg_target, cls_target
        )
        expected = 2.0 * seg_loss + 0.5 * cls_loss
        assert abs(total.item() - expected.item()) < 1e-5
    def test_differentiable(self) -> None:
        seg_out = torch.randn(1, 1, 16, 16, requires_grad=True)
        cls_out = torch.randn(1, 4, requires_grad=True)
        seg_target = torch.zeros(1, 1, 16, 16)
        cls_target = torch.tensor([0])
        total, _, _ = CombinedJointLoss()(seg_out, cls_out, seg_target, cls_target)
        total.backward()
        assert seg_out.grad is not None
        assert cls_out.grad is not None
