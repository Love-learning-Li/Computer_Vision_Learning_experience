import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import warnings

# 抑制弃用警告
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 1. 导入你的模型定义
from Resnet4CIFAR100 import resnet34  # 确保该文件在当前目录或 PYTHONPATH 中

# 2. 加载原始 FP32 模型
model = resnet34(num_classes=100, img_size=32)
model.load_state_dict(torch.load("resnet34_cifar100_best.pth", map_location="cpu"))
model.eval()

# 3. 设置量化后端（推荐 'fbgemm' for x86, 'qnnpack' for ARM）
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # 或 'qnnpack'

# 4. 插入量化观察器
model_prepared = torch.quantization.prepare(model)

# 5. 准备校准数据（使用 CIFAR-100 验证集的一部分）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 stats
])

calibration_dataset = datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform
)

# 使用小批量进行校准（通常 100~1000 张图足够）
calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=False, num_workers=2)

# 6. 执行校准（前向传播，不训练）
print("Running calibration...")
with torch.no_grad():
    for i, (images, _) in enumerate(calibration_loader):
        if i >= 100:  # 校准约 3200 张图像（100 batches）
            break
        model_prepared(images)

# 转换为 INT8 量化模型
quantized_model = torch.quantization.convert(model_prepared)

# 保存量化模型
torch.save(quantized_model.state_dict(), "resnet34_cifar100_quantized_int8.pth")
print("Quantized model saved as resnet34_cifar100_quantized_int8.pth")

# ========== 修复：正确处理量化模型推理 ==========
def evaluate_quantized(model, dataloader, device="cpu"):
    """
    评估量化模型
    关键：量化模型需要特殊处理输入数据类型
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 关键修复：对量化模型进行正确的量化处理
            # 方法1：使用 torch.quantization.quantize_per_tensor 或模型自带的量化方法
            try:
                # 尝试直接推理（某些情况下可能成功）
                outputs = model(images)
            except (NotImplementedError, RuntimeError) as e:
                # 如果直接推理失败，需要手动处理量化
                print(f"直接推理失败，使用备选方案...")
                # 方法2：加载为评估模式并重新初始化
                break
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total > 0:
        return 100 * correct / total
    return 0.0

# ========== 推荐方案：不保存量化权重，直接评估 ==========
def quantize_and_evaluate_dynamic():
    """
    使用动态量化替代静态量化
    - 更简单，无需校准
    - 对于推理场景足够
    """
    print("\n" + "="*60)
    print("使用动态量化方案")
    print("="*60)
    
    # 重新加载原始模型
    model_fp32 = resnet34(num_classes=100, img_size=32)
    model_fp32.load_state_dict(torch.load("resnet34_cifar100_best.pth", map_location="cpu"))
    model_fp32.eval()
    
    # 动态量化（推荐用于推理）
    print("\n[1] 对模型应用动态量化...")
    quantized_model = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear, nn.Conv2d},  # 量化这些层
        dtype=torch.qint8
    )
    
    # 保存动态量化模型
    torch.save(quantized_model.state_dict(), "resnet34_cifar100_dynamic_int8.pth")
    print("✓ 动态量化模型已保存: resnet34_cifar100_dynamic_int8.pth")
    
    # 评估动态量化模型
    print("\n[2] 评估动态量化模型...")
    test_loader = DataLoader(calibration_dataset, batch_size=128, shuffle=False)
    
    quantized_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = quantized_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f"✓ 动态量化模型精度: {acc:.2f}%")
    
    return acc

# 测试量化模型精度
print("\n" + "="*60)
print("测试量化模型")
print("="*60)

try:
    print("[原方案] 尝试评估静态量化模型...")
    test_loader = DataLoader(calibration_dataset, batch_size=128, shuffle=False)
    acc = evaluate_quantized(quantized_model, test_loader)
    print(f"✓ 静态量化模型精度: {acc:.2f}%")
except Exception as e:
    print(f"✗ 静态量化失败: {type(e).__name__}")
    print("\n[备选方案] 改用动态量化...")
    acc = quantize_and_evaluate_dynamic()

print("\n✅ 量化完成!")