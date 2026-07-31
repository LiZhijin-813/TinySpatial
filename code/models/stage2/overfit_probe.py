"""用于验证标签、损失、梯度和优化器链路的 BUS-only 过拟合模型。"""

import os

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor


class BUSOverfitProbe(nn.Module):
    """由单通道 TinyUSFM 编码器和线性四分类头组成的全可训练模型。"""

    def __init__(
        self,
        pretrained_path=None,
        num_classes=4,
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=12,
    ):
        """初始化全部默认可训练的 BUS-only 门禁模型。"""
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0,
            qkv_bias=True,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        """加载兼容的编码器权重，跳过切片嵌入与形状不匹配参数。"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise ValueError("预训练权重必须是参数字典或包含参数字典的检查点。")

        encoder_state = self.encoder.state_dict()
        compatible_state = {}
        prefixes = ("module.", "model.", "backbone.", "encoder.")
        for key, value in state_dict.items():
            name = key
            while name.startswith(prefixes):
                for prefix in prefixes:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                        break
            if "patch_embed" in name:
                continue
            if name in encoder_state and encoder_state[name].shape == value.shape:
                compatible_state[name] = value

        self.encoder.load_state_dict(compatible_state, strict=False)

    def forward(self, bus_img: Tensor) -> dict[str, Tensor]:
        """返回四分类 logits 与图像特征。"""
        encoder_features = self.encoder.forward_features(bus_img)
        if encoder_features.ndim == 2:
            image_features = encoder_features
        elif encoder_features.ndim == 3:
            image_features = encoder_features[:, 0]
        else:
            raise ValueError(
                "TinyUSFM 编码器特征必须是二维特征或三维 token 序列，"
                f"实际形状为 {tuple(encoder_features.shape)}。"
            )

        return {
            "class_logits": self.classifier(image_features),
            "image_features": image_features,
        }
