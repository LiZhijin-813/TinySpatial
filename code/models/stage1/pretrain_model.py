"""Stage 1: ACM-MIM 自监督预训练模型。

完整的前传流程：
    BUS+SWE → JointPatchEmbedding → MaskingEngine → TinyUSFM(可训练) → Decoder → Loss

此阶段 TinyUSFM 编码器完全可训练，目标是利用 SWE 提供的硬度信息
（如高杨氏模量点）来重建被掩码的 BUS 组织结构，强制建立"硬度-形态"
的深层物理关联。
"""
import os
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from .acm_mim import JointPatchEmbedding, MaskingEngine, LightweightDecoder, ReconstructionLoss


class ACMMIMPretrainModel(nn.Module):
    """Stage 1 ACM-MIM 自监督预训练模型。

    Pipeline: BUS+SWE → JointPatchEmbedding → MaskingEngine → TinyUSFM → Decoder → Loss

    TinyUSFM 在此阶段完全可训练 (requires_grad=True)。

    Attributes:
        patch_embed: 联合切片嵌入层 (4ch → patch tokens)
        encoder: TinyUSFM ViT 编码器 (12层, embed_dim=192)
        masking: 非对称掩码引擎 (75% BUS 掩码)
        decoder: 轻量级解码器 (2层 Transformer)
        recon_loss: 重建损失 (仅掩码位置 MSE)
    """

    def __init__(
        self,
        pretrained_path: str = None,
        embed_dim: int = 192,
        patch_size: int = 16,
        img_size: int = 224,
        depth: int = 12,
        num_heads: int = 12,
        mask_ratio: float = 0.75,
        decoder_dim: int = 96,
        decoder_depth: int = 2,
        decoder_num_heads: int = 6,
    ):
        """
        Args:
            pretrained_path: TinyUSFM 预训练权重路径
            embed_dim: 嵌入维度，默认 192
            patch_size: patch 尺寸，默认 16
            img_size: 输入图像尺寸，默认 224
            depth: 编码器层数，默认 12
            num_heads: 注意力头数，默认 12
            mask_ratio: 掩码比例，默认 0.75
            decoder_dim: 解码器维度，默认 96
            decoder_depth: 解码器层数，默认 2
            decoder_num_heads: 解码器注意力头数，默认 6
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        # 联合切片嵌入: BUS(1ch) + SWE(3ch) → 4ch → patch tokens
        self.patch_embed = JointPatchEmbedding(embed_dim, patch_size)

        # CLS token 与位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # TinyUSFM 骨干网络 (Stage 1 完全可训练)
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=4,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0,  # 仅输出特征，不含分类头
            qkv_bias=True,
        )
        # 替换编码器的 patch_embed 为联合版本
        self.encoder.patch_embed = self.patch_embed

        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)

        # 掩码引擎
        self.masking = MaskingEngine(mask_ratio)

        # 轻量级解码器
        self.decoder = LightweightDecoder(
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            patch_size=patch_size,
        )

        # 重建损失
        self.recon_loss = ReconstructionLoss(patch_size)

    def _load_pretrained(self, path: str):
        """加载 TinyUSFM 预训练权重，处理键名不匹配与形状差异。

        自动移除 'model.'/'module.'/'backbone.'/'encoder.' 前缀，
        跳过 patch_embed 权重（因 3ch→4ch 形状不匹配）。
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除常见前缀以匹配编码器 state_dict
            name = k.replace("model.", "").replace("module.", "").replace("backbone.", "").replace("encoder.", "")
            if name in self.encoder.state_dict():
                # 跳过 patch_embed 权重（3ch → 4ch 形状不匹配）
                if "patch_embed" in name:
                    continue
                new_state_dict[name] = v

        msg = self.encoder.load_state_dict(new_state_dict, strict=False)
        loaded = len(new_state_dict)
        total = len(self.encoder.state_dict())
        print(f"Loaded {loaded}/{total} encoder weights from {path}")
        if msg.missing_keys:
            print(f"  Missing keys: {[k for k in msg.missing_keys if 'patch_embed' not in k][:5]}")

    def forward(self, bus_img: torch.Tensor, swe_img: torch.Tensor):
        """Stage 1 完整前传。

        Args:
            bus_img: (B, 1, H, W) — 灰度 BUS 图像
            swe_img: (B, 3, H, W) — SWE 伪彩图像

        Returns:
            loss: 标量重建损失
            pred_pixels: (B, N, patch_size*patch_size*1) — 预测的 BUS 像素
            mask: (B, N) bool — 使用的掩码矩阵
        """
        B = bus_img.shape[0]
        device = bus_img.device

        # 联合切片嵌入
        tokens = self.patch_embed(bus_img, swe_img)  # (B, N, embed_dim)

        # 生成掩码
        mask, ids_restore = self.masking(self.num_patches, B, device)

        # 移除被掩码的 tokens，仅保留可见 tokens 送入编码器
        N = tokens.shape[1]
        len_keep = int(N * (1 - self.mask_ratio))

        # 重排：可见 token 在前，被掩码 token 在后
        ids_shuffle = torch.argsort(mask.float(), dim=1)
        ids_shuffle_visible = ids_shuffle[:, :len_keep]
        visible_tokens = torch.gather(
            tokens, 1, ids_shuffle_visible.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )

        # 重新推导恢复索引
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 前置 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        visible_tokens = torch.cat([cls_tokens, visible_tokens], dim=1)  # (B, 1+len_keep, embed_dim)

        # 添加位置编码（CLS + 各 patch）
        pos_embed_visible = torch.gather(
            self.pos_embed.expand(B, -1, -1),
            1,
            torch.cat([
                torch.zeros(B, 1, dtype=torch.long, device=device),  # CLS 位置
                ids_shuffle_visible + 1,  # patch 位置（偏移 1 以匹配 CLS）
            ], dim=1).unsqueeze(-1).expand(-1, -1, self.pos_embed.shape[-1]),
        )
        visible_tokens = visible_tokens + pos_embed_visible

        # 通过 TinyUSFM 编码器块（跳过内部 patch_embed/cls/pos，因为我们已自行处理）
        x = visible_tokens
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)

        # 移除 CLS token，送入解码器
        encoder_output = x[:, 1:, :]  # (B, len_keep, embed_dim)

        # 解码并重建
        pred_pixels = self.decoder(encoder_output, mask, ids_restore)

        # 仅在掩码位置计算损失
        loss = self.recon_loss(pred_pixels, bus_img, mask)

        return loss, pred_pixels, mask
