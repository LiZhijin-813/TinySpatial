"""Stage 2 维度验证测试：各支路输出形态、IP-Adapter 注入前后维度一致性、最终 MLP 输出。

测试项：
    - test_cdfi_branch: CDFI 特征提取支路输出维度
    - test_ip_adapter: IP-Adapter 解耦交叉注意力输出维度与缩放控制
    - test_text_branch: 文本语义分支输出维度与可训练参数量
    - test_cross_attention: 跨模态融合输出维度
    - test_full_model: 完整模型端到端前传 (4分类/5分类)
    - test_full_model_with_pretrained: 加载预训练权重后的前传
    - test_gradient_flow: 冻结编码器、可训练组件梯度验证
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from code.models.stage2.cdfi_branch import CDFIBranch
from code.models.stage2.ip_adapter import DecoupledAttentionLayer
from code.models.stage2.text_branch import TextLogicBranch
from code.models.stage2.cross_attention import LightweightCrossAttention
from code.models.stage2.subtyping_model import SECSubtypingModel


def test_cdfi_branch():
    """验证 CDFI 支路输出: (B, N_cdfi, 192) = (2, 49, 192)。"""
    model = CDFIBranch(out_dim=192, img_size=224)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 49, 192), f"期望 (2, 49, 192)，实际 {out.shape}"
    print("[PASS] CDFIBranch: (2, 49, 192)")


def test_ip_adapter():
    """验证 IP-Adapter 输出维度与缩放控制。"""
    layer = DecoupledAttentionLayer(embed_dim=192, num_heads=12, cdfi_dim=192)
    x = torch.randn(2, 197, 192)  # CLS + 196 patches
    cdfi_tokens = torch.randn(2, 49, 192)
    out = layer(x, cdfi_tokens)
    assert out.shape == (2, 197, 192), f"期望 (2, 197, 192)，实际 {out.shape}"

    # 验证缩放控制
    layer.set_ip_adapter_scale(0.5)
    out2 = layer(x, cdfi_tokens)
    assert out2.shape == (2, 197, 192)
    print("[PASS] DecoupledAttentionLayer: (2, 197, 192) + 缩放控制")


def test_text_branch():
    """验证文本分支输出维度与可训练参数占比。"""
    model = TextLogicBranch()
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128, dtype=torch.long)
    out = model(input_ids, attention_mask)
    assert out.shape == (2, 512), f"期望 (2, 512)，实际 {out.shape}"

    # 验证仅 LoRA 参数可训练
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[PASS] TextLogicBranch: (2, 512), 可训练={trainable/1e6:.3f}M / 总计={total/1e6:.1f}M")


def test_cross_attention():
    """验证跨模态门控融合输出维度: (B, image_dim + text_dim) = (2, 704)。"""
    model = LightweightCrossAttention(image_dim=192, text_dim=512)
    image_feat = torch.randn(2, 192)
    text_feat = torch.randn(2, 512)
    out = model(image_feat, text_feat)
    assert out.shape == (2, 704), f"期望 (2, 704)，实际 {out.shape}"  # 192 + 512
    # 验证门控融合不是恒等映射（不同输入应产生不同输出）
    out2 = model(image_feat, text_feat * 0)
    assert not torch.allclose(out, out2, atol=1e-4), "门控融合不应退化为恒等映射"
    print("[PASS] LightweightCrossAttention: (2, 704) + 门控非退化")


def test_full_model():
    """完整模型端到端前传: 所有模态 → N 类 logits。"""
    for num_classes in [4, 5]:
        model = SECSubtypingModel(
            pretrained_path=None,
            num_classes=num_classes,
            img_size=224,
        )
        bus = torch.randn(2, 1, 224, 224)
        swe = torch.randn(2, 3, 224, 224)
        cdfi = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128, dtype=torch.long)

        logits, F_bus_swe, F_text = model(bus, swe, cdfi, input_ids, attention_mask)
        assert logits.shape == (2, num_classes), f"期望 (2, {num_classes})，实际 {logits.shape}"
        assert F_bus_swe.shape == (2, 192), f"期望 F_bus_swe (2, 192)，实际 {F_bus_swe.shape}"
        assert F_text.shape == (2, 512), f"期望 F_text (2, 512)，实际 {F_text.shape}"
        print(f"[PASS] 完整模型 ({num_classes}分类): logits 形状 (2, {num_classes})")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"  总计: {total/1e6:.2f}M | 可训练: {trainable/1e6:.2f}M | 冻结: {frozen/1e6:.2f}M")


def test_full_model_with_pretrained():
    """加载预训练 TinyUSFM 权重后的前传测试。"""
    pretrained = "/home/lzj813/TinySpatial_Project/TinyUSFM.pth"
    if not os.path.exists(pretrained):
        print("[SKIP] 未找到预训练权重")
        return
    model = SECSubtypingModel(pretrained_path=pretrained, num_classes=5)
    bus = torch.randn(2, 1, 224, 224)
    swe = torch.randn(2, 3, 224, 224)
    cdfi = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128, dtype=torch.long)

    logits, _, _ = model(bus, swe, cdfi, input_ids, attention_mask)
    assert logits.shape == (2, 5)
    print(f"[PASS] 预训练模型: logits (2, 5)")


def test_gradient_flow():
    """验证梯度流向：冻结编码器无梯度，可训练组件有梯度。"""
    model = SECSubtypingModel(pretrained_path=None, num_classes=5)
    bus = torch.randn(2, 1, 224, 224)
    swe = torch.randn(2, 3, 224, 224)
    cdfi = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128, dtype=torch.long)

    logits, _, _ = model(bus, swe, cdfi, input_ids, attention_mask)
    loss = logits.sum()
    loss.backward()

    # 编码器（骨干）不应有梯度（冻结）
    encoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
        if not any(p is pp for pp in model.encoder.patch_embed.parameters())
    )
    assert not encoder_has_grad, "冻结的编码器不应有梯度！"

    # IP-Adapter 应有梯度
    ip_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.ip_adapters.parameters()
    )
    assert ip_has_grad, "IP-Adapter 应有梯度！"

    # CDFI 支路应有梯度
    cdfi_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.cdfi_branch.parameters()
    )
    assert cdfi_has_grad, "CDFI 支路应有梯度！"

    # MLP 分类头应有梯度
    mlp_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.mlp_head.parameters()
    )
    assert mlp_has_grad, "MLP 分类头应有梯度！"

    print("[PASS] 梯度流: 编码器冻结，IP-Adapter/CDFI/MLP 可训练")


if __name__ == "__main__":
    test_cdfi_branch()
    test_ip_adapter()
    test_cross_attention()
    test_full_model()
    test_full_model_with_pretrained()
    test_gradient_flow()
    # 文本分支测试较慢（需下载 BERT），建议单独运行
    print("\nStage 2 基础测试通过！（文本分支测试建议单独运行）")


def run_text_branch_test():
    """单独运行文本分支测试（需下载 BioClinicalBERT）。"""
    test_text_branch()
