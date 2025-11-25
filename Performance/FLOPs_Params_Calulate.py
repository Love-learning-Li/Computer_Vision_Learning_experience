import torch
from thop import profile
# 假设你有一个定义好的 MobileNet 模型实例
# from torchvision.models import mobilenet_v2
from Models.Resnet.Resnet4CIFAR100 import resnet18, resnet34

model = resnet34()
input_tensor = torch.randn(1, 3, 224, 224)

# 计算 MACs 和 Params
macs, params = profile(model, inputs=(input_tensor, ))

print(f"MACs: {macs / 1e9:.2f} G") # 输出为 Giga MACs
print(f"Params: {params / 1e6:.2f} M") # 输出为 Mega Params

# resnet18测试:
# MACs: 27.34 G
# Params: 11.22 M

# resnet34测试:
# MACs: 57.01 G
# Params: 21.33 M



