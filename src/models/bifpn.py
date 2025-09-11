import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv
class WeightedBiFPNLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_td = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn_td = nn.BatchNorm2d(channels)
        self.conv_bu = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn_bu = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, p_td, p_bu):
        if p_td.shape[1] != p_bu.shape[1]:
            proj_bu = nn.Conv2d(p_bu.shape[1], p_td.shape[1], kernel_size=1, bias=False).to(p_bu.device)
            p_bu = proj_bu(p_bu)
        p_td_fused = self.relu(self.bn_td(
            self.conv_td(p_td + F.interpolate(p_bu, size=p_td.shape[2:], mode="bilinear", align_corners=True))
        ))
        if p_td.shape[1] != p_bu.shape[1]:
            proj_td = nn.Conv2d(p_td.shape[1], p_bu.shape[1], kernel_size=1, bias=False).to(p_td.device)
            p_td = proj_td(p_td)
        p_bu_fused = self.relu(self.bn_bu(
            self.conv_bu(p_bu + F.interpolate(p_td, size=p_bu.shape[2:], mode="bilinear", align_corners=True))
        ))
        return p_td_fused, p_bu_fused
class BiFPNUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, num_cls_labels=4):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bifpn = WeightedBiFPNLayer(512)
        self.dconv_up4 = DoubleConv(1024 + 512, 512)
        self.dconv_up3 = DoubleConv(512 + 256, 256)
        self.dconv_up2 = DoubleConv(256 + 128, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(1024, num_cls_labels)
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        bottleneck = self.bottleneck(x)
        cls_logits = self.cls_head(self.avgpool(bottleneck).flatten(1))
        up_bottleneck = self.upsample(bottleneck)
        conv4_fused, _ = self.bifpn(conv4, up_bottleneck)
        x = self.dconv_up4(torch.cat([up_bottleneck, conv4_fused], dim=1))
        x = self.upsample(x)
        x = self.dconv_up3(torch.cat([x, conv3], dim=1))
        x = self.upsample(x)
        x = self.dconv_up2(torch.cat([x, conv2], dim=1))
        x = self.upsample(x)
        x = self.dconv_up1(torch.cat([x, conv1], dim=1))
        seg_out = self.seg_head(x)
        return seg_out, cls_logits