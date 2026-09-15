"""Small U-Net with a torchvision ResNet-18 encoder."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


class Up(nn.Module):
    def __init__(self, inputs, skip, outputs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inputs + skip, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class PetUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.up3 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up1 = Up(128, 64, 64)
        self.up0 = Up(64, 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        size = x.shape[-2:]
        stem = self.stem(x)
        x1 = self.layer1(self.pool(stem))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        decoded = self.up0(self.up1(self.up2(self.up3(x4, x3), x2), x1), stem)
        return F.interpolate(self.head(decoded), size=size, mode="bilinear", align_corners=False)
