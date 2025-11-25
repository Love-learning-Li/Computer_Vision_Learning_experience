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
# 1. Image100数据集加载（修正版）
# ----------------------------
def get_image100_loaders(batch_size, 
                        data_path="/root/exercise_prj/Attention_Learning/data/imagenet100",
                        train_split=0.8):
    """
    加载Image100数据集
    """
    data_path = Path(data_path)
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


    print(f"📁 检测到单一目录结构，将按 {train_split:.0%} 划分训练/测试集")
    full_dataset = torchvision.datasets.ImageFolder(root=str(data_path),
                                                    transform=None # 先不应用transform
        )
        
    # 计算划分大小
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    test_size = total_size - train_size
        
    # 生成确定性随机索引（避免 random_split(range(...)) 兼容性问题）
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(total_size, generator=g).tolist()
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]
        
    # 为 train/test 分别构造带独立 transform 的数据集，再按索引取子集
    train_base = torchvision.datasets.ImageFolder(root=str(data_path),
                                                  transform=transform_train)
    test_base = torchvision.datasets.ImageFolder(root=str(data_path),
                                                 transform=transform_test)
        
    train_dataset = torch.utils.data.Subset(train_base, train_idx)
    test_dataset = torch.utils.data.Subset(test_base, test_idx)

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
    num_classes = len(train_dataset.classes)
    
    
    return train_loader, test_loader, num_classes


# ----------------------------
# 2. Training & Evaluation
# ----------------------------
def compute_topk(outputs, targets, topk=(1, 5)):
    max_k = min(max(topk), outputs.size(1))
    _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        k = min(k, outputs.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.item())
    return res


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    top1_correct = 0
    top5_correct = 0
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
        top1, top5 = compute_topk(outputs, targets, topk=(1, 5))
        top1_correct += top1
        top5_correct += top5
    return (
        total_loss / len(loader),
        100. * correct / total,
        100. * top1_correct / total,
        100. * top5_correct / total
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    top1_correct = 0
    top5_correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            top1, top5 = compute_topk(outputs, targets, topk=(1, 5))
            top1_correct += top1
            top5_correct += top5
    end_time = time.time()
    once_delay_time = end_time - start_time
    return (
        total_loss / len(loader),
        100. * correct / total,
        100. * top1_correct / total,
        100. * top5_correct / total, 
        once_delay_time
    )


# ----------------------------
# 3. Main Training Loop
# ----------------------------
def main():
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ============================================================= #
    # Hyperparameters
    batch_size = 128
    epochs = 200
    lr = 5e-2
    warmup_epochs = min(5, max(0, epochs - 1))
    logger.info(f"超参数 - Batch: {batch_size}, Epochs: {epochs}, LR: {lr}, Warmup: {warmup_epochs}")
    # ============================================================= #


    save_dir = Path("Models/MobileNet/Pretrained")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "mobilenetv1_image100_best.pth"

    # 记录数据路径
    data_path = "/root/exercise_prj/Attention_Learning/data/imagenet100"
    logger.info(f"使用数据路径: {data_path}")

    # Data - 使用Image100数据集（自动获取类别数）
    logger.info("正在加载 Image100 数据集...")
    trainloader, testloader, num_classes = get_image100_loaders(
        batch_size=batch_size,
        data_path=data_path,
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
    best_top1k = 0.0
    avarge_delay_time = 0.0
    fast_delay_time = 999.9
    slow_delay_time = 0.0
    once_delay_time = 0.0
    all_delay_time = 0.0
    for epoch in range(epochs):
        train_loss, train_acc, train_top1, train_top5 = train_epoch(model, trainloader, criterion, optimizer, scheduler, device)
        test_loss, test_acc, test_top1, test_top5, once_delay_time = evaluate(model, testloader, criterion, device)
        all_delay_time += once_delay_time
        avarge_delay_time = all_delay_time / (epoch + 1)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch [{epoch + 1}/{epochs}] LR: {current_lr:.6f} | "
                   f"Train Loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Top-1: {train_top1:.2f}%, Top-5: {train_top5:.2f}% | \n"
                   f"Test Loss: {test_loss:.4f},Test acc: {test_acc:.4f}, Top-1: {test_top1:.2f}%, Top-5: {test_top5:.2f}% | \n"
                   f"Epoch [{epoch + 1}/{epochs}] , Test Once Delay: {once_delay_time:.4f}s, Avarge Delay: {avarge_delay_time:.4f}s | ")

        if test_top1 > best_top1k:
            best_top1k = test_top1
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ New best Top-1 accuracy: {best_top1k:.2f}% — model saved!")

        if once_delay_time < fast_delay_time:
            fast_delay_time = once_delay_time
            # torch.save(model.state_dict(), "resnet34_cifar100_best.pth")
            # print(f"✅ New best delay time : {fast_delay_time:.4f}%")
        
        if once_delay_time > slow_delay_time:
            slow_delay_time = once_delay_time
            # torch.save(model.state_dict(), "resnet34_cifar100_best.pth")
            # print(f"✅ New slow delay time: {slow_delay_time:.4f}%")

    logger.info("=" * 80)
    logger.info(f"🎉 Training finished. Best test Top-1 accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()


