"""Stage 2: SEC-Subtyping — 语义增强对比分型多模态微调架构。

完整前传流程：
    BUS+SWE → JointPatchEmbedding → 冻结 TinyUSFM（CDFI IP-Adapter 默认注入最后两层）→ F_bus_swe (B×192)
    CDFI → MobileNetV2 → 投影层 → CDFI Tokens → IP-Adapter 注入
    Text → BioClinicalBERT (冻结) → F_text (B×512)
    F_bus_swe + F_text → LightweightCrossAttention → (B, 192+512)
    → 配置驱动的平坦分类头或良恶性与亚型双任务头

核心设计：
    - TinyUSFM 骨干冻结，仅解冻最后 2 层，保留预训练表征
    - IP-Adapter 仅在最后 2 层注入 CDFI 血流信息，减少可训练参数
    - BioClinicalBERT 完全冻结，仅投影层可训练，避免文本分支过拟合
    - 任务头根据任务模式输出平坦分类 logits 或双任务 logits
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
from .task_heads import Stage2TaskHeads


class SECSubtypingModel(nn.Module):
    """Stage 2: 语义增强对比分型模型。

    多模态微调架构，将 BUS+SWE 解剖硬度特征、CDFI 血流特征、
    临床文本语义特征融合后，由配置驱动的平坦分类头或双任务头完成决策。

    Attributes:
        patch_embed: 联合切片嵌入层 (4ch → patch tokens)
        encoder: 冻结的 TinyUSFM ViT 编码器
        cdfi_branch: CDFI 特征提取支路 (MobileNetV2)
        ip_adapters: IP-Adapter 解耦交叉注意力层（默认最后两层）
        text_branch: 文本语义分支 (BioClinicalBERT 冻结 + 投影层)
        cross_attention: 跨模态特征融合模块
        task_heads: 平坦分类头或良恶性与亚型双任务头
    """

    def __init__(
        self,
        pretrained_path: str = None,
        task_mode: str = "flat4",
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
            task_mode: 任务模式，可选 flat4、flat5 或 dual_head，默认 flat4
            embed_dim: 嵌入维度，默认 192
            patch_size: patch 尺寸，默认 16
            img_size: 输入图像尺寸，默认 224
            depth: 编码器层数，必须为正整数，默认 12
            num_heads: 注意力头数，默认 12
            ip_adapter_layers: 注入 IP-Adapter 的层索引列表，默认编码器最后两层
            ip_adapter_scale: IP-Adapter 缩放因子，默认 0.1
            unfreeze_last_n: 解冻编码器最后 N 层，默认 2（保守微调，防止特征坍缩）
        """
        super().__init__()
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError("depth 必须是非布尔正整数")
        if (
            isinstance(unfreeze_last_n, bool)
            or not isinstance(unfreeze_last_n, int)
            or not 0 <= unfreeze_last_n <= depth
        ):
            raise ValueError("unfreeze_last_n 必须是 0 到 depth 的非布尔整数")

        if ip_adapter_layers is None:
            ip_adapter_layers = list(range(max(0, depth - 2), depth))
        else:
            try:
                ip_adapter_layers = list(ip_adapter_layers)
            except TypeError as exc:
                raise ValueError("ip_adapter_layers 必须是编码器层索引序列") from exc
        if any(
            not isinstance(layer, int)
            or isinstance(layer, bool)
            or not 0 <= layer < depth
            for layer in ip_adapter_layers
        ):
            raise ValueError("ip_adapter_layers 必须是 0 到 depth-1 的整数索引")
        if len(set(ip_adapter_layers)) != len(ip_adapter_layers):
            raise ValueError("ip_adapter_layers 不能包含重复索引")

        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.task_mode = task_mode
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

        # 冻结底层，仅解冻最后 N 层、归一化层与切片嵌入层。
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

        # ===== 3. IP-Adapter 层（默认注入最后两层） =====
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

        # ===== 6. 配置驱动的任务头 =====
        self.task_heads = Stage2TaskHeads(
            input_dim=embed_dim + 512,
            task_mode=task_mode,
        )

    def _load_pretrained(self, path: str):
        """加载 TinyUSFM 预训练权重（支持 Stage 1 或原始权重）。

        自动处理不同格式的 checkpoint（含 model/model_state_dict 键），
        仅加载名称存在、Tensor 类型且形状完全匹配的编码器权重。

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

        encoder_state = self.encoder.state_dict()
        compatible_state = {}
        skipped_shape_mismatch = []
        # 先收集所有可能的兼容键名映射，顶层键优先于 encoder. 前缀键。
        for k, v in state_dict.items():
            name = k.replace("model.", "").replace("module.", "").replace("backbone.", "").replace("encoder.", "")
            if name not in encoder_state or not isinstance(v, torch.Tensor):
                continue
            if encoder_state[name].shape != v.shape:
                skipped_shape_mismatch.append(name)
                continue
            # 顶层键优先，避免未训练的 encoder. 前缀键覆盖已训练顶层键。
            if not k.startswith("encoder.") or name not in compatible_state:
                compatible_state[name] = v

        msg = self.encoder.load_state_dict(compatible_state, strict=False)
        loaded = len(compatible_state)
        total = len(self.encoder.state_dict())
        missing_non_patch = [k for k in msg.missing_keys if "patch_embed" not in k]
        if skipped_shape_mismatch:
            print(f"已跳过 {len(skipped_shape_mismatch)} 个形状不匹配的预训练权重")
        print(f"已从 {path} 加载 {loaded}/{total} 个编码器权重")
        if missing_non_patch:
            print(f"  缺失的非切片嵌入权重: {missing_non_patch}")
        if msg.unexpected_keys:
            print(f"  未预期权重: {msg.unexpected_keys[:10]}")

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
            包含任务 logits、图像特征、文本特征和融合特征的字典。
            平坦任务返回 class_logits；双任务返回 malignancy_logits 与 subtype_logits。
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

        # === 任务感知决策 ===
        task_outputs = self.task_heads(fused)
        return {
            **task_outputs,
            "image_features": F_bus_swe,
            "text_features": F_text,
            "fused_features": fused,
        }
