import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if x1.shape[2:] != g1.shape[2:]:
            x1 = F.interpolate(x1, size=g1.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class AttUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, num_cls_labels=4):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dconv_up4 = DoubleConv(1024 + 512, 512)
        self.dconv_up3 = DoubleConv(512 + 256, 256)
        self.dconv_up2 = DoubleConv(256 + 128, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        self.att4 = AttentionBlock(1024, 512, 512)
        self.att3 = AttentionBlock(512, 256, 256)
        self.att2 = AttentionBlock(256, 128, 128)
        self.att1 = AttentionBlock(128, 64, 64)
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