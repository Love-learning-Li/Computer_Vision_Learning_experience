import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from torchvision.transforms import autoaugment
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from Models.Resnet.Resnet4CIFAR100 import resnet18, resnet34, resnet50, resnet101
from Models.MobileNet.MobileNet4CIFAR100 import MobileNetV1
import logging
from datetime import datetime
import argparse
import os
from Models.MobileNet.MobileNet4ImageNet100 import MobileNetV1_4ImageNet100

# ----------------------------
# 0. 配置日志
# ----------------------------
def setup_logger(log_dir="logs"):
    """设置日志记录器"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"image100_training_{timestamp}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志保存到: {log_file}")
    return logger

# ----------------------------
# 路径解析工具
# ----------------------------
def resolve_data_path(user_path: str | None) -> Path:
    """
    按优先级解析数据路径：
    1) 命令行/显式传入路径
    2) 脚本同目录下的 data/Image100
    3) 项目内的 /root/exercise_prj/Attention_Learning/data/Image100
    4) 当前工作目录下的 data/Image100
    """
    if user_path:
        p = Path(user_path).expanduser()
        if not p.is_absolute():
            # 相对路径相对于脚本目录解析，避免依赖运行时工作目录
            p = Path(__file__).resolve().parent.joinpath(p).resolve()
        if p.exists():
            return p

    candidates = [
        Path(__file__).resolve().parent / "data" / "imagenet100",
        Path("/root/exercise_prj/Attention_Learning/data/imagenet100"),
        Path.cwd() / "data" / "imagenet100",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "未找到数据集目录。请确认以下任一目录存在并包含类目录：\n"
        + "\n".join([f"- {str(c)}" for c in candidates])
        + "\n或使用 --data_path 显式指定数据集路径。期望目录结构示例：\n"
        "data_path/\n"
        " ├─ train/ (或所有类别直接放在 data_path 下)\n"
        " │   ├─ classA/\n"
        " │   └─ classB/\n"
        " └─ val/\n"
        "     ├─ classA/\n"
        "     └─ classB/\n"
    )

# ----------------------------
# 1. Image100数据集加载（修正版）
# ----------------------------
def get_image100_loaders(batch_size, data_path=None, train_split=0.8):
    """
    加载Image100数据集
    """
    data_path = resolve_data_path(data_path)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    # 训练数据增强
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.72, 1.0), ratio=(0.9, 1.1)),
        # transforms.Resize(256),
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.4),
    ])

    # 测试数据变换
    transform_test = transforms.Compose([
        transforms.Resize(224),
        # transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    data_path = Path(data_path)

    if not data_path.exists():
        # 冗余保护，正常情况 resolve_data_path 已经保证存在
        raise FileNotFoundError(f"未找到数据集目录: {data_path}")

    # 检查目录结构
    if (data_path / "train").exists() and (data_path / "val").exists():
        # 方式1: 已经划分好train/val
        print("📁 检测到train/val目录结构")
        train_dataset = torchvision.datasets.ImageFolder(
            root=str(data_path / "train"),
            transform=transform_train
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=str(data_path / "val"),
            transform=transform_test
        )
    else:
        # 方式2: 所有数据在同一目录，需要手动划分
        print(f"📁 检测到单一目录结构，将按 {train_split:.0%} 划分训练/测试集")
        full_dataset = torchvision.datasets.ImageFolder(
            root=str(data_path),
            transform=None  # 先不应用transform
        )
        
        # 计算划分大小
        total_size = len(full_dataset)
        train_size = int(total_size * train_split)
        test_size = total_size - train_size
        
        # 随机划分数据集
        train_indices, test_indices = random_split(
            range(total_size), 
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 创建子集
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices.indices)
        
        # 为子集应用不同的transform
        train_dataset.dataset.transform = transform_train
        test_dataset.dataset.transform = transform_test

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=20, 
        pin_memory=True,
        # 保持 worker 常驻
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True  # 丢弃最后不完整的batch
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=20, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    print(f"✅ 训练集样本数: {len(train_dataset)}")
    print(f"✅ 测试集样本数: {len(test_dataset)}")
    
    # 获取类别数
    if hasattr(train_dataset, 'classes'):
        num_classes = len(train_dataset.classes)
        print(f"✅ 类别数: {num_classes}")
    else:
        num_classes = len(train_dataset.dataset.classes)
        print(f"✅ 类别数: {num_classes}")
    
    return train_loader, test_loader, num_classes


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


# ----------------------------
# 3. Main Training Loop
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Image100数据集根目录")
    args = parser.parse_args()

    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    epochs = 200
    lr = 5e-2
    warmup_epochs = min(5, max(0, epochs - 1))
    logger.info(f"超参数 - Batch: {batch_size}, Epochs: {epochs}, LR: {lr}, Warmup: {warmup_epochs}")

    save_dir = Path("Models/MobileNet/Pretrained")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "mobilenetv1_image100_best.pth"

    # 解析与记录数据路径
    resolved_path = resolve_data_path(args.data_path)
    logger.info(f"使用数据路径: {resolved_path}")

    # Data - 使用Image100数据集（自动获取类别数）
    logger.info("正在加载 Image100 数据集...")
    trainloader, testloader, num_classes = get_image100_loaders(
        batch_size=batch_size,
        data_path=str(resolved_path),
        train_split=0.8
    )

    # Model - 使用检测到的类别数
    logger.info(f"正在初始化 MobileNetV1 模型（{num_classes}类）...")
    model = MobileNetV1_4ImageNet100(num_classes=num_classes, alpha=1.0).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=5e-4,
        nesterov=True
    )

    # 学习率调度器
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-3,
                    total_iters=warmup_epochs
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, epochs - warmup_epochs)
                )
            ],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    
    logger.info("开始训练...")
    logger.info("=" * 80)

    # Training
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, scheduler, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch [{epoch + 1}/{epochs}] LR: {current_lr:.6f} | "
                   f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                   f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            logger.info(f"✅ New best accuracy: {best_acc:.2f}% — model saved!")

    logger.info("=" * 80)
    logger.info(f"🎉 Training finished. Best test accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()


