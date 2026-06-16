"""对比对齐模块 (Contrastive Alignment Module, CAM)。

在 512 维潜空间内计算图像混合特征与文本向量的 InfoNCE 对比损失 L_align，
消除模型对文本关键词的"捷径学习"。

核心特性：
    - 图像侧 + 文本侧各一个双层 MLP 投影头，映射至 512 维潜空间
    - 标准 InfoNCE 对比损失（温度参数 τ 可配置）
    - 可选去偏 InfoNCE 变体（基于类别先验的校正因子 q_ij）
    - CAM 仅作为训练正则化项，不参与推理时特征融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAlignmentModule(nn.Module):
    """对比对齐模块：在 512 维潜空间对齐图像-文本特征。

    组件：
        - image_proj: Linear(image_dim, 512) — 将 F_bus_swe 投影到潜空间
        - text_proj: Linear(text_dim, 512) — 将 F_text 投影到潜空间
        - InfoNCE 损失（温度参数可配置）
        - 可选去偏 InfoNCE 变体（类别先验校正）
    """

    def __init__(self, image_dim: int = 192, text_dim: int = 512, proj_dim: int = 512, temperature: float = 0.07):
        """
        Args:
            image_dim: 图像特征输入维度，默认 192
            text_dim: 文本特征输入维度，默认 512
            proj_dim: 投影潜空间维度，默认 512
            temperature: InfoNCE 温度参数 τ，默认 0.07
        """
        super().__init__()
        self.temperature = temperature
        self.proj_dim = proj_dim

        # 图像侧投影头：双层 MLP
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        # 文本侧投影头：双层 MLP
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
        """
        Args:
            image_features: (B, image_dim) — F_bus_swe 图像特征
            text_features: (B, text_dim) — F_text 文本特征

        Returns:
            loss: 标量 InfoNCE 对比损失
            image_embeds: (B, proj_dim) — L2 归一化后的图像嵌入
            text_embeds: (B, proj_dim) — L2 归一化后的文本嵌入
        """
        # 投影并 L2 归一化
        image_embeds = F.normalize(self.image_proj(image_features), dim=-1)  # (B, 512)
        text_embeds = F.normalize(self.text_proj(text_features), dim=-1)    # (B, 512)

        loss = self.info_nce_loss(image_embeds, text_embeds)

        return loss, image_embeds, text_embeds

    def info_nce_loss(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """标准 InfoNCE 对比损失。

        L = -log(exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ))

        使用对称损失：image→text + text→image
        """
        # 相似度矩阵: (B, B)
        logits = (image_embeds @ text_embeds.T) / self.temperature

        # 对角线为正样本对
        B = image_embeds.shape[0]
        labels = torch.arange(B, device=logits.device)

        # 对称损失：图像→文本 + 文本→图像
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2

    def debiased_info_nce_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        class_labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """去偏 InfoNCE 损失：基于类别先验的校正因子 q_ij。

        同类样本对通过 q_ij=0 降低惩罚，防止"假阴性"惩罚，
        使相同亚型的病人在潜空间中自然聚类。

        L_debiased = -log(exp(sim(I_i, T_i)/τ) / (exp(sim(I_i, T_i)/τ) + Σ_{j≠i} q_ij * exp(sim(I_i, T_j)/τ)))

        Args:
            image_embeds: (B, proj_dim) — 归一化图像嵌入
            text_embeds: (B, proj_dim) — 归一化文本嵌入
            class_labels: (B,) — 类别标签，用于构建校正矩阵

        Returns:
            loss: 标量去偏对比损失
        """
        B = image_embeds.shape[0]
        sim = image_embeds @ text_embeds.T  # (B, B)
        sim = sim / self.temperature

        # 构建校正矩阵: q_ij = 0（同类），q_ij = 1（异类）
        if class_labels is not None:
            same_class = class_labels.unsqueeze(0) == class_labels.unsqueeze(1)  # (B, B)
            q = torch.ones_like(sim, dtype=torch.float)
            q[same_class] = 0.0
        else:
            q = torch.ones_like(sim, dtype=torch.float)

        # 正样本对相似度
        pos_sim = torch.diag(sim).unsqueeze(1)  # (B, 1)
        # 负样本对相似度（同类被置零）
        neg_sim = sim * q
        # 拼接为 logits: (B, 1+B)
        logits = torch.cat([pos_sim, neg_sim], dim=1)

        # 标签为 0（第一列为正样本）
        labels = torch.zeros(B, dtype=torch.long, device=sim.device)
        loss = F.cross_entropy(logits, labels)

        return loss
