import torch
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)
class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, num_cls_labels=4):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.dconv_down5 = DoubleConv(512, 1024)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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
        bottleneck = self.dconv_down5(x)
        cls_logits = self.cls_head(self.avgpool(bottleneck).flatten(1))
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