"""Loss functions for segmentation and joint tasks.

- ``DiceLoss`` — soft Dice loss for binary / multi-class segmentation.
- ``CombinedJointLoss`` — weighted sum of Dice + CrossEntropy for joint training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss (1 - Dice coefficient).

    Supports both binary (sigmoid) and multi-class (softmax) segmentation.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] > 1:
            # Multi-class
            pred = F.softmax(pred, dim=1)
            target_one_hot = F.one_hot(
                target.long(), num_classes=pred.shape[1]
            ).permute(0, 3, 1, 2).float()
            target = target_one_hot
        else:
            pred = torch.sigmoid(pred)

        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return (1 - dice).mean()


class CombinedJointLoss(nn.Module):
    """Weighted combination of DiceLoss + CrossEntropyLoss for joint training.

    Parameters
    ----------
    seg_weight : float
        Weight for the segmentation (Dice) loss component.
    cls_weight : float
        Weight for the classification (CrossEntropy) loss component.
    """

    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 1.0) -> None:
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        seg_out: torch.Tensor,
        cls_out: torch.Tensor,
        seg_target: torch.Tensor,
        cls_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(total_loss, seg_loss, cls_loss)``."""
        loss_seg = self.dice_loss(seg_out, seg_target)
        loss_cls = self.ce_loss(cls_out, cls_target)
        return (
            self.seg_weight * loss_seg + self.cls_weight * loss_cls,
            loss_seg,
            loss_cls,
        )
