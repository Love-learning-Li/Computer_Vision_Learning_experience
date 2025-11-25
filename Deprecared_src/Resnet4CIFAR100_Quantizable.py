"""
量化友好的 ResNet 实现
专门为 PyTorch 量化优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional

class BasicBlock(nn.Module):
    """ResNet 基础残差块（量化友好版本）"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)  # 量化需要 inplace=False
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = nn.ReLU(inplace=False)
        
        # 用于残差连接的量化加法
        self.skip_add = FloatFunctional()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 使用量化友好的加法
        out = self.skip_add.add(out, identity)
        out = self.relu2(out)
        return out
    
    def fuse_model(self):
        """融合 Conv + BN + ReLU"""
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1']], inplace=True)
        torch.quantization.fuse_modules(self, [['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, [['0', '1']], inplace=True)

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
    """ResNet 主干网络（量化友好版本）"""
    def __init__(self, block, layers, in_channels=3):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
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
    """完整的 ResNet 分类器（量化友好版本）"""
    def __init__(self, block, layers, num_classes=100, in_channels=3, 
                 img_size=32):
        super().__init__()
        
        # 量化/反量化存根
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        self.backbone = ResNetBackbone(block, layers, in_channels)
        
        # 针对小图像的调整（如 CIFAR-10/100）
        if img_size <= 32:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, 
                                          stride=1, padding=1, bias=False)
            # 移除 maxpool，使用跳过连接
            self.use_maxpool = False
        else:
            self.use_maxpool = True

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.quant(x)  # 量化输入
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        
        if self.use_maxpool:
            x = self.backbone.maxpool(x)

        x = self.backbone.stage1(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        x = self.dequant(x)  # 反量化输出
        return x
    
    def fuse_model(self):
        """融合模块以优化推理"""
        # 融合初始层
        torch.quantization.fuse_modules(self.backbone, 
            [['conv1', 'bn1', 'relu']], inplace=True)
        
        # 融合每个残差块
        for stage in [self.backbone.stage1, self.backbone.stage2, 
                     self.backbone.stage3, self.backbone.stage4]:
            for block in stage.blocks:
                block.fuse_model()

def resnet18(num_classes=100, img_size=32):
    """ResNet-18（量化友好版本）"""
    return ResNetClassifier(BasicBlock, [2, 2, 2, 2], num_classes, img_size=img_size)

def resnet34(num_classes=100, img_size=32):
    """ResNet-34（量化友好版本）"""
    return ResNetClassifier(BasicBlock, [3, 4, 6, 3], num_classes, img_size=img_size)
