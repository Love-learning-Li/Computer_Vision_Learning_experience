"""
ResNet-18 量化处理 - 简化版示例
展示最常用的静态量化方法
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from Resnet4CIFAR100_Quantizable import resnet18  # 使用量化友好版本
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ============================================================================
# 步骤 1: 准备数据
# ============================================================================
def get_data_loader(batch_size=128):
    """获取 CIFAR-100 测试数据"""
    CIFAR100_MEAN = (0.507075, 0.486548, 0.440917)
    CIFAR100_STD = (0.267334, 0.256438, 0.276150)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    
    dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

# ============================================================================
# 步骤 2: 配置量化
# ============================================================================
def prepare_model_for_quantization(model):
    """
    配置模型量化参数
    'fbgemm': x86 CPU 后端（Intel/AMD）
    'qnnpack': ARM CPU 后端（移动设备）
    """
    # 设置量化配置
    model.qconfig = quant.get_default_qconfig('fbgemm')
    
    print("量化配置:", model.qconfig)
    
    # 融合模块（Conv + BN + ReLU）
    model.fuse_model()
    print("模块融合完成")
    
    # 准备模型：插入观察器
    model = quant.prepare(model, inplace=False)
    
    return model

# ============================================================================
# 步骤 3: 校准模型
# ============================================================================
def calibrate(model, data_loader, num_batches=100):
    """
    运行校准过程：通过数据收集激活值统计信息
    
    参数:
        model: 准备好的模型（已插入观察器）
        data_loader: 校准数据加载器
        num_batches: 使用多少批次数据（通常 100-200 足够）
    """
    model.eval()
    print(f"\n开始校准，使用 {num_batches} 个批次...")
    
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            # 前向传播，观察器会自动收集统计信息
            model(images)
            
            if (i + 1) % 20 == 0:
                print(f"  校准进度: {i+1}/{num_batches}")
    
    print("校准完成！")
    return model

# ============================================================================
# 步骤 4: 转换为量化模型
# ============================================================================
def convert_to_quantized(model):
    """
    将准备好的模型转换为量化模型
    权重和激活值从 FP32 转换为 INT8
    """
    quantized_model = quant.convert(model, inplace=False)
    print("\n模型已转换为量化版本")
    return quantized_model

# ============================================================================
# 步骤 5: 评估性能
# ============================================================================
def evaluate(model, data_loader, device='cpu'):
    """评估模型准确率"""
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# ============================================================================
# 主流程
# ============================================================================
def main():
    print("="*70)
    print("ResNet-18 静态量化 - 完整流程演示")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    data_loader = get_data_loader()
    
    # 2. 加载模型
    print("\n[2/6] 加载 ResNet-18 模型...")
    model = resnet18(num_classes=100, img_size=32)
    model.eval()
    
    # 可选：加载预训练权重
    # model.load_state_dict(torch.load('resnet18_cifar100.pth'))
    
    # 3. 评估原始模型
    print("\n[3/6] 评估原始 FP32 模型...")
    acc_original = evaluate(model, data_loader)
    print(f"原始模型准确率: {acc_original:.2f}%")
    
    # 4. 准备量化
    print("\n[4/6] 配置量化参数...")
    model = prepare_model_for_quantization(model)
    
    # 5. 校准
    print("\n[5/6] 校准模型...")
    model = calibrate(model, data_loader, num_batches=100)
    
    # 6. 转换并评估
    print("\n[6/6] 转换为量化模型并评估...")
    quantized_model = convert_to_quantized(model)
    
    acc_quantized = evaluate(quantized_model, data_loader)
    print(f"量化模型准确率: {acc_quantized:.2f}%")
    print(f"准确率下降: {acc_original - acc_quantized:.2f}%")
    
    # 保存模型
    torch.save(quantized_model, 'resnet18_quantized.pth')
    print(f"\n量化模型已保存到: resnet18_quantized.pth")
    
    print("\n" + "="*70)
    print("量化完成！")
    print("="*70)
    print("\n模型大小对比:")
    print(f"  原始模型 (FP32): ~42 MB")
    print(f"  量化模型 (INT8): ~11 MB (减少约 75%)")
    
    print("\n使用量化模型进行推理:")
    print("  quantized_model = torch.load('resnet18_quantized.pth')")
    print("  output = quantized_model(input)")

if __name__ == '__main__':
    main()
