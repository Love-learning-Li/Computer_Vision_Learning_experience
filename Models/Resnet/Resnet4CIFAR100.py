import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ResNet 基础残差块（用于 ResNet-18/34）"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        # Height = (W - K + 2*P) / S + 1 
        # Width  = (W - K + 2*P) / S + 1 
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        # Height = (W - K + 2*P) / S + 1 = 32
        # Width  = (W - K + 2*P) / S + 1 = 32
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    """ResNet 瓶颈块（用于 ResNet-50/101/152）"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out

class ResStage(nn.Module):
    """ResNet 残差阶段"""
    def __init__(self, block, in_channels, out_channels, num_blocks, stride=1):
        super().__init__()
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(in_channels, out_channels))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)

class ResNetBackbone(nn.Module):
    """ResNet 主干网络"""
    def __init__(self, block, layers, in_channels=3):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层（类似 ViT 的 patch embedding）
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        # Height = (W - K + 2*P) / S + 1 = (32-7+6)/2+1 = 16.5 = 16
        # Width  = (W - K + 2*P) / S + 1 = (32-7+6)/2+1 = 16.5 = 16
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差阶段
        self.stage1 = ResStage(block, 64, 64, layers[0])
        self.stage2 = ResStage(block, 64 * block.expansion, 128, layers[1], stride=2)
        self.stage3 = ResStage(block, 128 * block.expansion, 256, layers[2], stride=2)
        self.stage4 = ResStage(block, 256 * block.expansion, 512, layers[3], stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class ResNetClassifier(nn.Module):
    """完整的 ResNet 分类器"""
    def __init__(self, block, layers, num_classes=100, in_channels=3, 
                 img_size=32):  # 为 CIFAR-100 设计
        super().__init__()
        self.backbone = ResNetBackbone(block, layers, in_channels)
        
        # 针对小图像的调整（如 CIFAR-10/100）
        if img_size <= 32:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, 
                                          stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()  # 移除 maxpool

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.backbone(x)  # (B, 512*expansion)
        x = self.fc(x)        # (B, num_classes)
        return x

def resnet18(num_classes=100, img_size=32):
    """ResNet-18"""
    return ResNetClassifier(BasicBlock, [2, 2, 2, 2], num_classes, img_size=img_size)

def resnet34(num_classes=100, img_size=32):
    """ResNet-34"""
    return ResNetClassifier(BasicBlock, [3, 4, 6, 3], num_classes, img_size=img_size)

def resnet50(num_classes=100, img_size=32):
    """ResNet-50"""
    return ResNetClassifier(Bottleneck, [3, 4, 6, 3], num_classes, img_size=img_size)

def resnet101(num_classes=100, img_size=32):
    """ResNet-101"""
    return ResNetClassifier(Bottleneck, [3, 4, 23, 3], num_classes, img_size=img_size)


