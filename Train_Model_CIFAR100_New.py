import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from torchvision.transforms import autoaugment
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from Models.Resnet.Resnet4CIFAR100 import resnet18, resnet34, resnet50, resnet101
from Models.MobileNet.MobileNet4CIFAR100 import MobileNetV1
# from Attention4CIFAR_New import ViT_CIFAR
# from Attention4CIFAR100New import HybridVisionTransformer

# ----------------------------
# 1. CIAFR-100数据集加载
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


# ----------------------------
# 3. Main Training Loop
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    epochs = 200
    lr = 3e-2

    # 创建保存目录（如果不存在）
    save_dir = Path("Models/MobileNet/Pretrained")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "mobilenetv1_cifar100_best.pth"

    # Data
    trainloader, testloader = get_cifar100_loaders(batch_size)

    # Model
    model = MobileNetV1(num_classes=100, alpha=1.0).to(device)
    model = model.cuda()

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 定义优化器 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # 定义余弦退火学习率调度器
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    

    # Training
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, scheduler, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best accuracy: {best_acc:.2f}% — model saved!")

    print(f"🎉 Training finished. Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()