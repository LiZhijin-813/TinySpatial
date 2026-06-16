"""CDFI 特征提取管线：MobileNetV2 → 投影层 → CDFI Tokens。

使用剥离分类层的 MobileNetV2 提取 CDFI 血流图像的空间特征图，
再经投影层映射为与 TinyUSFM 维度对齐的 CDFI Tokens。

特征管线：
    CDFI 图像 (B, 3, H, W)
    → MobileNetV2 (无分类层) → 空间特征图 (B, 1280, H', W')
    → 展平 + 投影层 → CDFI Tokens (B, N_cdfi, 192)
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class CDFIBranch(nn.Module):
    """CDFI 血流特征提取支路。

    使用 MobileNetV2 (~2.0M 参数) 提取 CDFI 图像的空间特征图，
    再经投影层映射为与 TinyUSFM 维度对齐的 CDFI Tokens。
    这些 CDFI Tokens 将与 [CLS] token 及 BUS+SWE patch tokens
    一同进入 TinyUSFM 前传，在第 6-11 层通过 IP-Adapter 注入血流信息。
    """

    def __init__(self, out_dim: int = 192, img_size: int = 224):
        """
        Args:
            out_dim: 输出维度，需与 TinyUSFM embed_dim 对齐，默认 192
            img_size: 输入图像尺寸，默认 224
        """
        super().__init__()
        # MobileNetV2 特征提取器（移除分类层），输出: (B, 1280, 7, 7)
        backbone = mobilenet_v2(weights=None)
        self.features = backbone.features

        # 从特征图空间尺寸计算 N_cdfi
        self.feat_h = img_size // 32  # MobileNetV2 将 224→7
        self.feat_w = img_size // 32
        self.cnn_out_channels = 1280
        self.n_cdfi = self.feat_h * self.feat_w  # 49

        # 投影层：将每个空间位置从 1280 维映射到 192 维
        self.projection = nn.Sequential(
            nn.Linear(self.cnn_out_channels, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, cdfi_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cdfi_img: (B, 3, H, W) — CDFI 彩色多普勒血流图

        Returns:
            cdfi_tokens: (B, N_cdfi, out_dim=192) — CDFI 特征 token 序列
        """
        feat = self.features(cdfi_img)  # (B, 1280, H', W')
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # (B, H'*W', 1280) 展平空间维度
        cdfi_tokens = self.projection(feat)      # (B, N_cdfi, 192) 投影到对齐维度
        return cdfi_tokens
