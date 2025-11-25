import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from Attention4Mnist import AttentionMNISTClassifier

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device:",device)


# ----------------------------
# 1. MNIST数据集加载
# ----------------------------
def get_mnist_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值和标准差
    ])

    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)

    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ----------------------------
# 2. Training & Evaluation
# ----------------------------
def train_epoch(model, loader, criterion, optimizer, device):
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
    batch_size = 64
    epochs = 50
    lr = 1e-3

    # Data
    trainloader, testloader = get_mnist_loaders(batch_size)

    # Model
    model = AttentionMNISTClassifier().to(device)
    model = model.cuda()

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        # scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "bnn_cifar10_best.pth")
            print(f"✅ New best accuracy: {best_acc:.2f}% — model saved!")

    print(f"🎉 Training finished. Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()