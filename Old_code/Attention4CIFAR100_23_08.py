import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """将图像划分为 patches 并嵌入到向量空间"""
    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )  # 直接用卷积实现 patch embedding
        # 在 __init__ 中添加
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H//patch, W//patch)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)

        x = attn @ v  # (B, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim=None, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        if mlp_hidden_dim is None:
            mlp_hidden_dim = embed_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AttentionCifarClassifier_100(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_channels=3,
                 embed_dim=256, num_heads=8, num_layers=12, num_classes=100):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        # GAP方式实现
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码长度 +1（为 [class] token 预留）
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        # x = self.patch_embed(x)  # (B, N, embed_dim)
        # x = x + self.pos_embed  # 加位置编码
        # x = self.dropout(x)
        # x = self.transformer_blocks(x)  # (B, N, embed_dim)
        # =======================================================
        # 引入CLS位置编码因子
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # === 关键改动：添加 [class] token ===
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, embed_dim)
        
        x = x + self.pos_embed  # (B, N+1, embed_dim) ← [class] token 也有位置编码
        x = self.dropout(x)

        x = self.transformer_blocks(x)  # (B, N+1, embed_dim)

        # === 关键改动：取 [class] token 的输出 ===
        x = x[:, 0]  # (B, embed_dim) ← 只取 [class] token，保留动态聚合信息

        x = self.norm(x)
        x = self.classifier(x)  # (B, num_classes)
        return x

model = AttentionCifarClassifier_100()