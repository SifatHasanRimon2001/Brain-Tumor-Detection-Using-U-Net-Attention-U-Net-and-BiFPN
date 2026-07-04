"""Attention U-Net — U-Net with additive attention gates on skip connections.

Based on "Attention U-Net: Learning Where to Look for the Pancreas"
(Oktay et al., 2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.components import AttentionBlock, DoubleConv


class AttentionUNet(nn.Module):
    """U-Net augmented with attention gates on skip connections.

    Parameters
    ----------
    in_ch : int
        Number of input channels (default 3).
    num_classes : int
        Number of segmentation output channels (1 for binary).
    num_cls_labels : int
        Number of classification labels.
    """

    def __init__(
        self,
        in_ch: int = 3,
        num_classes: int = 1,
        num_cls_labels: int = 4,
    ) -> None:
        super().__init__()

        # ── Encoder ────────────────────────────────────────────────
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        # ── Attention gates ────────────────────────────────────────
        self.att4 = AttentionBlock(1024, 512, 512)
        self.att3 = AttentionBlock(512, 256, 256)
        self.att2 = AttentionBlock(256, 128, 128)
        self.att1 = AttentionBlock(128, 64, 64)

        # ── Decoder ────────────────────────────────────────────────
        self.dconv_up4 = DoubleConv(1024 + 512, 512)
        self.dconv_up3 = DoubleConv(512 + 256, 256)
        self.dconv_up2 = DoubleConv(256 + 128, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        # ── Heads ──────────────────────────────────────────────────
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(1024, num_cls_labels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        bottleneck = self.bottleneck(x)

        # Classification
        cls_logits = self.cls_head(self.avgpool(bottleneck).flatten(1))

        # Decoder with attention-gated skip connections
        x = self.upsample(bottleneck)
        x = self.dconv_up4(torch.cat([x, self.att4(x, conv4)], dim=1))

        x = self.upsample(x)
        x = self.dconv_up3(torch.cat([x, self.att3(x, conv3)], dim=1))

        x = self.upsample(x)
        x = self.dconv_up2(torch.cat([x, self.att2(x, conv2)], dim=1))

        x = self.upsample(x)
        x = self.dconv_up1(torch.cat([x, self.att1(x, conv1)], dim=1))

        seg_out = self.seg_head(x)
        return seg_out, cls_logits
