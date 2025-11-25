"""
ResNet-18 量化处理完整示例
支持三种量化方式：
1. 动态量化 (Dynamic Quantization) - 最简单，仅量化权重
2. 静态量化 (Static Quantization) - 需要校准数据，量化权重和激活
3. 量化感知训练 (Quantization Aware Training, QAT) - 训练时模拟量化
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.quantization as quant
from Resnet4CIFAR100 import resnet18
import copy
import time
import os

# CIFAR-100 数据集参数
CIFAR100_TRAIN_MEAN = (0.507075, 0.486548, 0.440917)
CIFAR100_TRAIN_STD  = (0.267334, 0.256438, 0.276150)

def get_cifar100_loaders(batch_size=128, num_workers=2):
    """获取 CIFAR-100 数据加载器"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def evaluate_model(model, test_loader, device='cpu'):
    """评估模型准确率和推理速度"""
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    end_time = time.time()
    accuracy = 100. * correct / total
    inference_time = end_time - start_time
    
    return accuracy, inference_time

def get_model_size(model):
    """获取模型大小（MB）"""
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / 1e6
    os.remove("temp.pth")
    return size

# ============================================================================
# 方法 1: 动态量化 (最简单，适合 LSTM/Transformer)
# ============================================================================
def dynamic_quantization(model):
    """
    动态量化：仅量化权重，激活值在推理时动态量化
    优点：实现简单，不需要校准数据
    缺点：对 CNN 效果不如静态量化
    """
    print("\n" + "="*60)
    print("方法 1: 动态量化 (Dynamic Quantization)")
    print("="*60)
    
    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # 要量化的层类型
        dtype=torch.qint8        # 量化为 8-bit 整数
    )
    
    return quantized_model

# ============================================================================
# 方法 2: 静态量化 (推荐用于 CNN)
# ============================================================================
def prepare_model_for_static_quantization(model):
    """
    准备模型进行静态量化
    需要插入观察器来收集统计信息
    """
    # 设置量化配置
    model.qconfig = quant.get_default_qconfig('fbgemm')  # x86 CPU 使用 'fbgemm'
    
    # 融合模块以提高效率 (Conv + BN + ReLU)
    model = quant.fuse_modules(model, [
        ['backbone.conv1', 'backbone.bn1', 'backbone.relu']
    ])
    
    # 准备量化：插入观察器
    quant.prepare(model, inplace=True)
    
    return model

def calibrate_model(model, calibration_loader, device='cpu'):
    """
    校准模型：通过少量数据收集激活值的统计信息
    这一步是静态量化的关键
    """
    model.eval()
    model.to(device)
    
    print("开始校准模型...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= 100:  # 通常 100-200 个 batch 足够
                break
            images = images.to(device)
            model(images)
            if (i + 1) % 20 == 0:
                print(f"  校准进度: {i+1}/100 batches")
    
    print("校准完成！")
    return model

def static_quantization(model, calibration_loader, device='cpu'):
    """
    静态量化：量化权重和激活值
    优点：推理速度快，模型小
    缺点：需要校准数据
    """
    print("\n" + "="*60)
    print("方法 2: 静态量化 (Static Quantization)")
    print("="*60)
    
    # 1. 准备模型
    model_prepared = prepare_model_for_static_quantization(model)
    
    # 2. 校准模型（收集统计信息）
    model_prepared = calibrate_model(model_prepared, calibration_loader, device)
    
    # 3. 转换为量化模型
    quantized_model = quant.convert(model_prepared, inplace=False)
    
    return quantized_model

# ============================================================================
# 方法 3: 量化感知训练 (QAT) - 精度最高
# ============================================================================
def prepare_qat_model(model):
    """准备模型进行量化感知训练"""
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    
    # 融合模块
    model = quant.fuse_modules(model, [
        ['backbone.conv1', 'backbone.bn1', 'backbone.relu']
    ])
    
    # 准备 QAT
    quant.prepare_qat(model, inplace=True)
    
    return model

def train_qat_model(model, train_loader, test_loader, epochs=5, device='cuda'):
    """
    量化感知训练
    在训练过程中模拟量化效果
    """
    print("\n" + "="*60)
    print("方法 3: 量化感知训练 (QAT)")
    print("="*60)
    
    model = model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {running_loss/100:.4f}, Acc: {100.*correct/total:.2f}%")
                running_loss = 0.0
        
        # 测试
        acc, _ = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1} 测试准确率: {acc:.2f}%")
    
    # 转换为量化模型
    model.eval()
    model.to('cpu')
    quantized_model = quant.convert(model, inplace=False)
    
    return quantized_model

# ============================================================================
# 主函数：演示三种量化方法
# ============================================================================
def main():
    import os
    
    print("ResNet-18 量化处理演示")
    print("="*60)
    
    # 加载数据
    print("\n加载 CIFAR-100 数据集...")
    train_loader, test_loader = get_cifar100_loaders(batch_size=128)
    
    # 加载预训练模型（如果有）
    print("\n加载 ResNet-18 模型...")
    model_fp32 = resnet18(num_classes=100, img_size=32)
    
    # 尝试加载预训练权重
    if os.path.exists('resnet18_cifar100.pth'):
        print("加载预训练权重...")
        model_fp32.load_state_dict(torch.load('resnet18_cifar100.pth'))
    else:
        print("警告：未找到预训练权重，使用随机初始化的模型")
        print("      量化效果可能不佳，建议先训练模型")
    
    # 评估原始模型
    print("\n" + "="*60)
    print("评估原始 FP32 模型")
    print("="*60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    acc_fp32, time_fp32 = evaluate_model(model_fp32, test_loader, device)
    print(f"FP32 模型准确率: {acc_fp32:.2f}%")
    print(f"FP32 推理时间: {time_fp32:.2f}秒")
    
    # ========================================================================
    # 方法 1: 动态量化
    # ========================================================================
    model_dynamic = copy.deepcopy(model_fp32)
    model_dynamic = dynamic_quantization(model_dynamic)
    
    acc_dynamic, time_dynamic = evaluate_model(model_dynamic, test_loader, 'cpu')
    print(f"动态量化准确率: {acc_dynamic:.2f}%")
    print(f"动态量化推理时间: {time_dynamic:.2f}秒")
    print(f"准确率下降: {acc_fp32 - acc_dynamic:.2f}%")
    
    # ========================================================================
    # 方法 2: 静态量化
    # ========================================================================
    model_static = copy.deepcopy(model_fp32)
    model_static.eval()
    
    # 使用训练集的一部分作为校准数据
    calibration_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )
    
    model_static = static_quantization(model_static, calibration_loader, 'cpu')
    
    acc_static, time_static = evaluate_model(model_static, test_loader, 'cpu')
    print(f"静态量化准确率: {acc_static:.2f}%")
    print(f"静态量化推理时间: {time_static:.2f}秒")
    print(f"准确率下降: {acc_fp32 - acc_static:.2f}%")
    
    # ========================================================================
    # 方法 3: QAT (可选，需要较长时间)
    # ========================================================================
    run_qat = input("\n是否运行量化感知训练 (QAT)? 这需要较长时间 (y/n): ")
    if run_qat.lower() == 'y':
        model_qat = copy.deepcopy(model_fp32)
        model_qat = prepare_qat_model(model_qat)
        
        device_qat = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_qat = train_qat_model(model_qat, train_loader, test_loader, 
                                    epochs=3, device=device_qat)
        
        acc_qat, time_qat = evaluate_model(model_qat, test_loader, 'cpu')
        print(f"QAT 量化准确率: {acc_qat:.2f}%")
        print(f"QAT 量化推理时间: {time_qat:.2f}秒")
        print(f"准确率下降: {acc_fp32 - acc_qat:.2f}%")
    
    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "="*60)
    print("量化结果总结")
    print("="*60)
    print(f"{'方法':<20} {'准确率':<15} {'推理时间':<15} {'准确率下降'}")
    print("-"*60)
    print(f"{'FP32 (原始)':<20} {acc_fp32:>6.2f}%{'':<8} {time_fp32:>6.2f}秒{'':<8} {'N/A'}")
    print(f"{'动态量化':<20} {acc_dynamic:>6.2f}%{'':<8} {time_dynamic:>6.2f}秒{'':<8} {acc_fp32-acc_dynamic:>6.2f}%")
    print(f"{'静态量化':<20} {acc_static:>6.2f}%{'':<8} {time_static:>6.2f}秒{'':<8} {acc_fp32-acc_static:>6.2f}%")
    
    print("\n量化建议:")
    print("1. 动态量化：适合 RNN/LSTM，对 CNN 效果有限")
    print("2. 静态量化：CNN 推荐，准确率和速度平衡最好")
    print("3. QAT：准确率要求高时使用，但训练时间长")
    
    # 保存量化模型
    torch.save(model_static, 'resnet18_cifar100_quantized.pth')
    print("\n已保存静态量化模型到: resnet18_cifar100_quantized.pth")

if __name__ == '__main__':
    main()
