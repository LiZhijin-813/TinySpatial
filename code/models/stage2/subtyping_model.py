"""Stage 2: SEC-Subtyping — 语义增强对比分型多模态微调架构。

完整前传流程：
    BUS+SWE → JointPatchEmbedding → 冻结 TinyUSFM (+ CDFI IP-Adapter at 第10-11层) → F_bus_swe (B×192)
    CDFI → MobileNetV2 → 投影层 → CDFI Tokens → IP-Adapter 注入
    Text → BioClinicalBERT (冻结) → F_text (B×512)
    F_bus_swe + F_text → LightweightCrossAttention → (B, 192+512)
    → MLP → N 类 softmax

核心设计：
    - TinyUSFM 骨干冻结，仅解冻最后 2 层，保留预训练表征
    - IP-Adapter 仅在最后 2 层注入 CDFI 血流信息，减少可训练参数
    - BioClinicalBERT 完全冻结，仅投影层可训练，避免文本分支过拟合
    - MLP head 加大 dropout，防止小样本过拟合
"""
import os

# 配置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from ..stage1.acm_mim import JointPatchEmbedding
from .cdfi_branch import CDFIBranch
from .ip_adapter import DecoupledAttentionLayer
from .text_branch import TextLogicBranch
from .cross_attention import LightweightCrossAttention


class SECSubtypingModel(nn.Module):
    """Stage 2: 语义增强对比分型模型。

    多模态微调架构，将 BUS+SWE 解剖硬度特征、CDFI 血流特征、
    临床文本语义特征融合后进行 N 分类决策（4分类或5分类含良性）。

    Attributes:
        patch_embed: 联合切片嵌入层 (4ch → patch tokens)
        encoder: 冻结的 TinyUSFM ViT 编码器
        cdfi_branch: CDFI 特征提取支路 (MobileNetV2)
        ip_adapters: IP-Adapter 解耦交叉注意力层 (第 10-11 层)
        text_branch: 文本语义分支 (BioClinicalBERT 冻结 + 投影层)
        cross_attention: 跨模态特征融合模块
        mlp_head: N 分类 MLP 决策头
    """

    def __init__(
        self,
        pretrained_path: str = None,
        num_classes: int = 4,
        embed_dim: int = 192,
        patch_size: int = 16,
        img_size: int = 224,
        depth: int = 12,
        num_heads: int = 12,
        ip_adapter_layers: list = None,
        ip_adapter_scale: float = 0.1,
        unfreeze_last_n: int = 2,
    ):
        """
        Args:
            pretrained_path: TinyUSFM 预训练权重路径
            num_classes: 分类类别数，默认 4 (4分类: Luminal A/B, HER2+, TNBC; 5分类: +Benign)
            embed_dim: 嵌入维度，默认 192
            patch_size: patch 尺寸，默认 16
            img_size: 输入图像尺寸，默认 224
            depth: 编码器层数，默认 12
            num_heads: 注意力头数，默认 12
            ip_adapter_layers: 注入 IP-Adapter 的层索引列表，默认 [10,11]
            ip_adapter_scale: IP-Adapter 缩放因子，默认 0.1
            unfreeze_last_n: 解冻编码器最后 N 层，默认 2（保守微调，防止特征坍缩）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        if ip_adapter_layers is None:
            ip_adapter_layers = list(range(10, 12))
        self.ip_adapter_layers = ip_adapter_layers

        # ===== 1. BUS+SWE 分支: JointPatchEmbedding + TinyUSFM =====
        self.patch_embed = JointPatchEmbedding(embed_dim, patch_size)

        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=4,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0,
            qkv_bias=True,
        )
        # 替换编码器的 patch_embed 为联合版本
        self.encoder.patch_embed = self.patch_embed

        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)

        # Partial fine-tuning: 冻结底层，仅解冻最后 N 层 + norm + patch_embed
        for param in self.encoder.parameters():
            param.requires_grad = False
        # 解冻最后 N 层
        for i in range(depth - unfreeze_last_n, depth):
            for param in self.encoder.blocks[i].parameters():
                param.requires_grad = True
        # 解冻最终 LayerNorm 和 patch_embed
        for param in self.encoder.norm.parameters():
            param.requires_grad = True
        for param in self.encoder.patch_embed.parameters():
            param.requires_grad = True

        # ===== 2. CDFI 分支 =====
        self.cdfi_branch = CDFIBranch(out_dim=embed_dim, img_size=img_size)

        # ===== 3. IP-Adapter 层（注入第 10-11 层） =====
        self.ip_adapters = nn.ModuleDict({
            str(i): DecoupledAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, cdfi_dim=embed_dim)
            for i in ip_adapter_layers
        })
        for adapter in self.ip_adapters.values():
            adapter.set_ip_adapter_scale(ip_adapter_scale)

        # ===== 4. 文本分支 =====
        self.text_branch = TextLogicBranch()

        # ===== 5. 跨模态特征融合 =====
        self.cross_attention = LightweightCrossAttention(
            image_dim=embed_dim,  # 192
            text_dim=512,
        )

        # ===== 6. 分类 MLP =====
        # 输入: F_bus_swe(192) + cross_attn_out(512) = 704
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 512),
            nn.Linear(embed_dim + 512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def _load_pretrained(self, path: str):
        """加载 TinyUSFM 预训练权重（支持 Stage 1 或原始权重）。

        自动处理不同格式的 checkpoint（含 model/model_state_dict 键），
        跳过 patch_embed 权重（3ch→4ch 形状不匹配）。

        Stage 1 checkpoint 中同时存在顶层和 encoder. 前缀的 pos_embed/cls_token，
        顶层键（训练过的）优先于 encoder. 前缀键（VisionTransformer 原始未训练值）。
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # 先收集所有可能的键名映射，顶层键优先于 encoder. 前缀键
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("model.", "").replace("module.", "").replace("backbone.", "").replace("encoder.", "")
            if name in self.encoder.state_dict():
                if "patch_embed" in name:
                    continue  # 跳过: 3ch→4ch 形状不匹配
                # 顶层键（不含 encoder. 前缀的原始键）优先，避免未训练值覆盖已训练值
                if not k.startswith("encoder."):
                    new_state_dict[name] = v
                elif name not in new_state_dict:
                    new_state_dict[name] = v

        msg = self.encoder.load_state_dict(new_state_dict, strict=False)
        loaded = len(new_state_dict)
        total = len(self.encoder.state_dict())
        missing_non_patch = [k for k in msg.missing_keys if "patch_embed" not in k]
        print(f"Loaded {loaded}/{total} encoder weights from {path}")
        if missing_non_patch:
            print(f"  Missing (non-patch_embed): {missing_non_patch}")
        if msg.unexpected_keys:
            print(f"  Unexpected: {msg.unexpected_keys[:10]}")

    def set_ip_adapter_scale(self, scale: float):
        """全局设置所有 IP-Adapter 层的缩放因子。

        Args:
            scale: 缩放因子
        """
        for adapter in self.ip_adapters.values():
            adapter.set_ip_adapter_scale(scale)

    def forward(self, bus_img, swe_img, cdfi_img, input_ids, attention_mask):
        """Stage 2 完整前传。

        Args:
            bus_img: (B, 1, H, W) — 灰度 BUS 图像
            swe_img: (B, 3, H, W) — SWE 伪彩图像
            cdfi_img: (B, 3, H, W) — CDFI 彩色多普勒血流图
            input_ids: (B, max_len) — 临床文本 token ID
            attention_mask: (B, max_len) — 注意力掩码

        Returns:
            logits: (B, num_classes) — N 分类 logits
            F_bus_swe: (B, embed_dim) — 图像混合特征（用于 CAM）
            F_text: (B, text_dim) — 文本特征（用于 CAM）
        """
        B = bus_img.shape[0]

        # === 提取 CDFI tokens ===
        cdfi_tokens = self.cdfi_branch(cdfi_img)  # (B, N_cdfi, 192)

        # === BUS+SWE patch tokens ===
        tokens = self.patch_embed(bus_img, swe_img)  # (B, N, 192)

        # 前置 CLS token
        cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1+N, 192)

        # 添加位置编码
        tokens = tokens + self.encoder.pos_embed

        # === 逐层前传，指定层注入 IP-Adapter ===
        x = tokens
        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)  # 标准自注意力 + FFN

            # 在指定层注入 CDFI 血流信息
            if i in self.ip_adapter_layers or str(i) in self.ip_adapters:
                key = str(i)
                if key in self.ip_adapters:
                    x = self.ip_adapters[key](x, cdfi_tokens)

        x = self.encoder.norm(x)

        # 提取 F_bus_swe：CLS + 全局平均池化，避免 CLS 坍缩
        cls_feat = x[:, 0, :]              # (B, 192) — CLS token
        gap_feat = x[:, 1:, :].mean(dim=1)  # (B, 192) — patch tokens 均值
        F_bus_swe = cls_feat + gap_feat      # (B, 192) — 互补融合

        # === 文本特征 ===
        F_text = self.text_branch(input_ids, attention_mask)  # (B, 512)

        # === 跨模态特征融合 ===
        fused = self.cross_attention(F_bus_swe, F_text)  # (B, 192+512)

        # === N 分类决策 ===
        logits = self.mlp_head(fused)  # (B, num_classes)

        return logits, F_bus_swe, F_text
