"""跨模态门控融合模块：图像特征 + 文本特征 → 融合表示。

原始交叉注意力在单 token 输入下退化为恒等映射（softmax(1 key)=1.0），
改用门控融合机制：学习一个动态门控信号控制文本信息注入量，
避免信息退化为简单的恒等传递。
"""
import torch
import torch.nn as nn


class LightweightCrossAttention(nn.Module):
    """跨模态门控融合模块。

    以图像特征为主干，文本特征为调制信号，通过可学习门控动态
    控制文本信息注入量。输出为 [图像特征; 门控融合输出]。

    设计要点：
        - 门控机制：g = sigmoid(W_g[img;txt] + b_g)，自适应调节融合比例
        - 拼接而非相加：保留图像原始表征，文本信息作为补充信号
        - 残差连接：确保文本分支的梯度能直接回传
    """

    def __init__(self, image_dim: int = 192, text_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim

        # 文本特征投影到 image_dim，用于门控融合
        self.text_proj = nn.Linear(text_dim, image_dim)

        # 门控信号：从拼接特征计算动态融合权重
        self.gate = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.Sigmoid(),
        )

        # 融合后投影到 text_dim 输出
        self.fuse_proj = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, text_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: (B, image_dim) — F_bus_swe
            text_features: (B, text_dim) — F_text

        Returns:
            fused: (B, image_dim + text_dim) — [图像特征; 门控融合输出]
        """
        # 投影文本到图像维度
        text_proj = self.text_proj(text_features)  # (B, image_dim)

        # 计算门控信号
        g = self.gate(torch.cat([image_features, text_proj], dim=-1))  # (B, image_dim)

        # 门控融合：g * text_proj + (1 - g) * image_features
        fused_img = g * text_proj + (1 - g) * image_features  # (B, image_dim)

        # 投影到 text_dim 输出
        fused_out = self.fuse_proj(fused_img)  # (B, text_dim)

        # 拼接图像特征与融合输出
        return torch.cat([image_features, fused_out], dim=-1)  # (B, image_dim + text_dim)
