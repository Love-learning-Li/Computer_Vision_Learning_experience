import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.quantization import QuantStub, DeQuantStub, fuse_modules

# 1. 量化友好的 ResNet 模块
class QuantizableBasicBlock(nn.Module):
    """量化友好的基础残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)  # 注意：inplace=False 用于量化
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
        
        # 量化工具
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.add_relu.add_relu(out, identity)  # 量化友好的 add + relu
        return out

class QuantizableResNetBackbone(nn.Module):
    """量化友好的 ResNet 主干"""
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        # 初始层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)  # inplace=False 用于量化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个阶段
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 量化工具
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)  # 量化输入
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        x = self.dequant(x)  # 反量化输出
        return x

def quantizable_resnet18(num_classes=1000):
    """创建量化友好的 ResNet-18"""
    return QuantizableResNetBackbone(QuantizableBasicBlock, [2, 2, 2, 2], num_classes)

# 2. 模型融合（提高推理效率）
def fuse_model(model):
    """融合 Conv + BN + ReLU 以提高量化效果"""
    # 融合主干网络
    fuse_modules(model, ['conv1', 'bn1', 'relu'], inplace=True)
    
    # 融合每个阶段
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            if hasattr(block, 'conv1') and hasattr(block, 'bn1') and hasattr(block, 'relu1'):
                fuse_modules(block, ['conv1', 'bn1', 'relu1'], inplace=True)
                fuse_modules(block, ['conv2', 'bn2'], inplace=True)
    
    return model

# 3. 量化准备和执行
def prepare_model_for_quantization(model):
    """准备模型进行量化"""
    # 融合模块
    model = fuse_model(model)
    
    # 设置为评估模式
    model.eval()
    
    # 配置量化方案
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 准备量化
    torch.quantization.prepare(model, inplace=True)
    
    return model

def quantize_model(model, calibration_loader, num_batches=10):
    """执行量化（使用校准数据）"""
    model.eval()
    
    print("开始校准量化...")
    with torch.no_grad():
        for i, (data, _) in enumerate(calibration_loader):
            if i >= num_batches:  # 只用前几个 batch 进行校准
                break
            model(data)
    
    # 转换为量化模型
    quantized_model = torch.quantization.convert(model, inplace=True)
    print("量化完成！")
    
    return quantized_model


# ----------------------------
# 1. CIAFR-10数据集加载
# ----------------------------
def get_cifar100_loaders(batch_size):

    CIFAR100_TRAIN_MEAN = (0.507075, 0.486548, 0.440917)
    CIFAR100_TRAIN_STD  = (0.267334, 0.256438, 0.276150)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])

    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ToTensor(),
    #     transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    # ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)

    return train_loader, test_loader


# ----------------------------
# 2. Training & Evaluation
# ----------------------------
def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), 100. * correct / total

# 4. 完整的量化流程示例
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    epochs = 200
    lr = 0.01

    # Data
    trainloader, testloader = get_cifar100_loaders(batch_size)
    
    # 创建量化友好的模型
    model = quantizable_resnet18(num_classes=100)
    
    # 准备量化
    model = prepare_model_for_quantization(model)
    
    # 创建模拟的校准数据（实际使用时应使用真实数据）
    
    
    # 这里用随机数据模拟校准数据
    dummy_data = test_dataset  # batch_size=16
    calibration_loader = [(dummy_data, torch.randint(0, 100, (16,))) for _ in range(5)]
    
    # 执行量化
    quantized_model = quantize_model(model, calibration_loader)
    
    # 测试量化模型
    test_input = test_dataset
    with torch.no_grad():
        output = quantized_model(test_input)
        print(f"量化模型输出形状: {output.shape}")
        print(f"量化模型大小: {torch.jit.trace(quantized_model, test_input).size()} bytes")
    
    # 与原始模型对比
    original_model = quantizable_resnet18(num_classes=100)
    original_size = sum(p.numel() for p in original_model.parameters()) * 4  # 假设 float32
    quantized_size = sum(p.numel() for p in quantized_model.parameters())  # int8
    print(f"原始模型参数量: {original_size} bytes")
    print(f"量化模型参数量: {quantized_size} bytes")
    print(f"压缩比: {original_size / quantized_size:.2f}x")





if __name__ == "__main__":
    main()



