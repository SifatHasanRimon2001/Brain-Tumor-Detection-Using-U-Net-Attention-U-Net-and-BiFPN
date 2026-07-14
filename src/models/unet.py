"""Standard U-Net architecture for segmentation + classification.
Based on the original U-Net paper with batch normalisation and
a classification head attached to the bottleneck.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from src.models.components import DoubleConv
class UNet(nn.Module):
    """U-Net with an optional classification head.
    Parameters
    ----------
    in_ch : int
        Number of input channels (default 3 for RGB).
    num_classes : int
        Number of segmentation output channels (1 for binary).
    num_cls_labels : int
        Number of classification labels (4 for BRISC tumour types).
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
        # Classification logits (from bottleneck)
        cls_logits = self.cls_head(self.avgpool(bottleneck).flatten(1))
        # Decoder with skip connections
        x = self.upsample(bottleneck)
        x = self.dconv_up4(torch.cat([x, conv4], dim=1))
        x = self.upsample(x)
        x = self.dconv_up3(torch.cat([x, conv3], dim=1))
        x = self.upsample(x)
        x = self.dconv_up2(torch.cat([x, conv2], dim=1))
        x = self.upsample(x)
        x = self.dconv_up1(torch.cat([x, conv1], dim=1))
        seg_out = self.seg_head(x)
        return seg_out, cls_logits
