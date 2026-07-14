"""BiFPN-Enhanced U-Net.
Replaces the simple decoder skip with a weighted bi-directional
feature pyramid network (BiFPN) at the deepest skip level.
**Fix**: The original implementation created ``nn.Conv2d`` layers
*dynamically inside the forward pass*, meaning they were never
registered as model parameters and produced random results each run.
Here we properly declare all projections in ``__init__``.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import DoubleConv
class WeightedBiFPNLayer(nn.Module):
    """A single top-down + bottom-up BiFPN fusion layer at a fixed channel count."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_td = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_td = nn.BatchNorm2d(channels)
        self.conv_bu = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_bu = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(
        self,
        p_td: torch.Tensor,
        p_bu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse top-down (*p_td*) and bottom-up (*p_bu*) features.
        Both inputs must have the same number of channels.
        """
        # Top-down fusion: upsample bottom-up to match top-down spatial size
        p_td_fused = self.relu(
            self.bn_td(
                self.conv_td(
                    p_td + F.interpolate(p_bu, size=p_td.shape[2:],
                                         mode="bilinear", align_corners=True)
                )
            )
        )
        # Bottom-up fusion: downsample top-down to match bottom-up spatial size
        p_bu_fused = self.relu(
            self.bn_bu(
                self.conv_bu(
                    p_bu + F.interpolate(p_td_fused, size=p_bu.shape[2:],
                                         mode="bilinear", align_corners=True)
                )
            )
        )
        return p_td_fused, p_bu_fused
class BiFPNUNet(nn.Module):
    """U-Net with a BiFPN fusion layer at the deepest skip connection.
    The 1024-channel bottleneck is projected to 512 channels before
    being passed into the BiFPN layer alongside the 512-channel
    encoder feature ``conv4``.
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
        # ── BiFPN projection & fusion ──────────────────────────────
        self.proj_to_bifpn = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.bifpn = WeightedBiFPNLayer(512)
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
        # BiFPN fusion at level 4 (skip + upsampled bottleneck)
        up_bottleneck = self.upsample(bottleneck)                   # 1024 ch
        proj_bottleneck = self.proj_to_bifpn(up_bottleneck)         # → 512 ch
        conv4_fused, _ = self.bifpn(conv4, proj_bottleneck)
        # Decoder with fused features
        x = self.dconv_up4(torch.cat([up_bottleneck, conv4_fused], dim=1))
        x = self.upsample(x)
        x = self.dconv_up3(torch.cat([x, conv3], dim=1))
        x = self.upsample(x)
        x = self.dconv_up2(torch.cat([x, conv2], dim=1))
        x = self.upsample(x)
        x = self.dconv_up1(torch.cat([x, conv1], dim=1))
        seg_out = self.seg_head(x)
        return seg_out, cls_logits
