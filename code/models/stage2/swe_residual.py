"""将 SWE 硬度图作为受控残差注入 BUS 图像特征。"""

import torch
import torch.nn as nn


class SWEResidualBranch(nn.Module):
    """轻量提取 SWE 特征，并用有界系数控制注入幅度。"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, bus_features: torch.Tensor, swe_img: torch.Tensor):
        """返回 BUS 特征与有界 SWE 残差之和。"""
        return bus_features + self.gate_logit.sigmoid() * self.features(swe_img)
