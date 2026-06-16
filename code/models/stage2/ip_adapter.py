"""IP-Adapter：解耦交叉注意力层，用于 CDFI 血流信息注入。

核心机制：在自注意力旁路并行添加注入分支（非重写自注意力）。
保持原有 K/V 映射锁定不变，新增可训练的 Wk_cdfi、Wv_cdfi
投影 CDFI Tokens。编码器内部所有 tokens 作为 Query，CDFI Tokens
作为 Key/Value，使 [CLS] token 逐层吸收血流信息。
"""
import torch
import torch.nn as nn


class DecoupledAttentionLayer(nn.Module):
    """IP-Adapter 解耦交叉注意力层。

    在自注意力输出的旁路并行添加 CDFI 交叉注意力分支，
    输出为 x + scale * IP-Adapter(x, cdfi_tokens)。

    设计要点：
        - 原有 K/V 映射锁定（冻结），不破坏已学到的 BUS-SWE 表征
        - 新增 Wk_cdfi、Wv_cdfi 可训练线性投影层
        - 主干 Q 与固有 K、新增 K_cdfi 分别计算内积，结果并行相加
        - 通过 set_ip_adapter_scale() 接口动态调节血流特征话语权
    """

    def __init__(self, embed_dim: int = 192, num_heads: int = 12, cdfi_dim: int = 192):
        """
        Args:
            embed_dim: 编码器嵌入维度，默认 192
            num_heads: 注意力头数，默认 12
            cdfi_dim: CDFI token 维度，默认 192（与 embed_dim 对齐）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # IP-Adapter 可训练投影层：将 CDFI tokens 投影为 K、V
        self.Wk_cdfi = nn.Linear(cdfi_dim, embed_dim, bias=False)
        self.Wv_cdfi = nn.Linear(cdfi_dim, embed_dim, bias=False)

        # IP-Adapter 分支的输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # IP-Adapter 缩放因子：控制血流特征注入强度
        self.ip_adapter_scale = 1.0

        # 残差分支零初始化：训练初期 IP-Adapter 输出接近 0，不破坏编码器特征
        nn.init.normal_(self.out_proj.weight, std=0.01)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def set_ip_adapter_scale(self, scale: float):
        """设置 IP-Adapter 缩放因子，动态调节血流特征话语权。

        Args:
            scale: 缩放因子，0.0 表示完全关闭注入，1.0 表示完全注入
        """
        self.ip_adapter_scale = scale

    def forward(self, x: torch.Tensor, cdfi_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim) — 自注意力输出（已计算完成）
            cdfi_tokens: (B, N_cdfi, cdfi_dim) — CDFI 特征 tokens

        Returns:
            (B, N, embed_dim) — x + scale * IP-Adapter 交叉注意力输出
        """
        B, N, _ = x.shape
        N_cdfi = cdfi_tokens.shape[1]

        # Q 来自编码器 tokens（自注意力输出），K/V 来自 CDFI tokens
        # IP-Adapter: Q=编码器 tokens, K=Wk_cdfi(CDFI), V=Wv_cdfi(CDFI)
        Q = x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, N, D)
        K = self.Wk_cdfi(cdfi_tokens).reshape(B, N_cdfi, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv_cdfi(cdfi_tokens).reshape(B, N_cdfi, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        scale = self.head_dim ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale  # (B, H, N, N_cdfi)
        attn = attn.softmax(dim=-1)

        ip_out = (attn @ V).transpose(1, 2).reshape(B, N, self.embed_dim)  # (B, N, embed_dim)
        ip_out = self.out_proj(ip_out)

        # 并行相加：自注意力输出 + 缩放后的 IP-Adapter 输出
        return x + self.ip_adapter_scale * ip_out
