import torch
import torch.nn as nn

class Attention4MNISTModel(nn.Module):
    """基于注意力机制的MNIST分类模型"""

    def __init__(self, img_size=28, num_classes=10, d_model=64, num_heads=4, num_layers=1):
        super(AttentionMNISTModel, self).__init__()
        self.img_size = img_size  # MNIST图像尺寸28×28
        self.d_model = d_model  # 特征维度（需能被num_heads整除，64÷4=16）
        self.num_layers = num_layers  # 注意力层数

        # 1. 输入处理：将28×28图像转换为序列并映射到d_model维度
        # 图像形状：(batch_size, 1, 28, 28) → 展平为序列：(batch_size, 28, 28)（每行作为一个token）
        self.input_proj = nn.Linear(img_size, d_model)  # 将每行28个像素映射到d_model维度

        # 2. 堆叠注意力层和前馈层
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attention': AttentionSublayer(d_model, num_heads),
                'ffn': FeedForwardSublayer(d_model, hidden_dim=128)  # 隐藏层维度设小些（MNIST简单）
            }))

        # 3. 分类头：将序列特征聚合后映射到10类
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # 输入x形状：(batch_size, 1, 28, 28)
        batch_size = x.size(0)

        # 步骤1：输入处理为序列
        x = x.squeeze(1)  # 去除通道维度：(batch_size, 28, 28)
        x = self.input_proj(x)  # 映射到d_model维度：(batch_size, 28, 64)

        # 步骤2：通过注意力层和前馈层
        attn_weights_list = []  # 保存注意力权重（可选，用于可视化）
        for layer in self.layers:
            x, attn_weights = layer['attention'](x)  # 注意力层
            attn_weights_list.append(attn_weights)
            x = layer['ffn'](x)  # 前馈层

        # 步骤3：特征聚合（取序列的平均作为全局特征）
        x = x.mean(dim=1)  # (batch_size, 64)

        # 步骤4：分类
        x = self.classifier(x)  # (batch_size, 10)

        return x, attn_weights_list  # 返回预测结果和注意力权重（可选）