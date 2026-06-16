"""Stage 1 维度验证测试：输入 (B,4,H,W) → 编码器 → 解码器 → 损失标量。

测试项：
    - test_joint_patch_embedding: 联合切片嵌入输出维度
    - test_masking_engine: 掩码引擎输出与比例
    - test_decoder: 轻量级解码器输出维度
    - test_reconstruction_loss: 重建损失为标量且为正
    - test_full_model_forward: 完整模型端到端前传
    - test_full_model_with_pretrained: 加载预训练权重后的前传
    - test_gradient_flow: 梯度回传至编码器参数
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from code.models.stage1.acm_mim import JointPatchEmbedding, MaskingEngine, LightweightDecoder, ReconstructionLoss
from code.models.stage1.pretrain_model import ACMMIMPretrainModel


def test_joint_patch_embedding():
    """验证联合切片嵌入输出维度: (B, N, embed_dim) = (2, 196, 192)。"""
    embed = JointPatchEmbedding(embed_dim=192, patch_size=16)
    bus = torch.randn(2, 1, 224, 224)
    swe = torch.randn(2, 3, 224, 224)
    out = embed(bus, swe)
    assert out.shape == (2, 196, 192), f"期望 (2, 196, 192)，实际 {out.shape}"
    print("[PASS] JointPatchEmbedding: (2,196,192)")


def test_masking_engine():
    """验证掩码引擎输出形状与掩码比例接近 75%。"""
    engine = MaskingEngine(mask_ratio=0.75)
    mask, ids_restore = engine(num_patches=196, batch_size=2, device=torch.device("cpu"))
    assert mask.shape == (2, 196)
    assert mask.dtype == torch.bool
    # 约 75% 应被掩码
    ratio = mask.float().mean().item()
    assert 0.7 < ratio < 0.8, f"掩码比例 {ratio:.2f} 偏离 0.75"
    print(f"[PASS] MaskingEngine: mask_ratio={ratio:.3f}")


def test_decoder():
    """验证轻量级解码器输出维度: (B, N, patch_size*patch_size*1)。"""
    decoder = LightweightDecoder(embed_dim=192, decoder_dim=96, depth=2, num_heads=6, patch_size=16)
    visible_tokens = torch.randn(2, 49, 192)  # 25% of 196 = 49 个可见 token
    mask = torch.zeros(2, 196, dtype=torch.bool)
    mask[:, 49:] = True  # 75% 被掩码
    ids_restore = torch.arange(196).unsqueeze(0).expand(2, -1)
    pred = decoder(visible_tokens, mask, ids_restore)
    assert pred.shape == (2, 196, 256), f"期望 (2, 196, 256)，实际 {pred.shape}"
    print("[PASS] LightweightDecoder: (2, 196, 256)")


def test_reconstruction_loss():
    """验证重建损失为标量且为正值。"""
    loss_fn = ReconstructionLoss(patch_size=16)
    pred = torch.randn(2, 196, 256)
    bus = torch.randn(2, 1, 224, 224)
    mask = torch.zeros(2, 196, dtype=torch.bool)
    mask[:, 49:] = True
    loss = loss_fn(pred, bus, mask)
    assert loss.dim() == 0, f"损失应为标量，实际形状 {loss.shape}"
    assert loss.item() > 0, "损失应为正值"
    print(f"[PASS] ReconstructionLoss: loss={loss.item():.4f}")


def test_full_model_forward():
    """完整模型端到端前传: (B,1,H,W)+(B,3,H,W) → loss 标量。"""
    model = ACMMIMPretrainModel(
        pretrained_path=None,  # 测试时跳过预训练权重
        embed_dim=192,
        patch_size=16,
        img_size=224,
        mask_ratio=0.75,
    )
    bus = torch.randn(2, 1, 224, 224)
    swe = torch.randn(2, 3, 224, 224)
    loss, pred, mask = model(bus, swe)
    assert loss.dim() == 0, f"损失应为标量，实际 {loss.shape}"
    assert pred.shape == (2, 196, 256), f"预测形状: {pred.shape}"
    assert mask.shape == (2, 196)
    print(f"[PASS] 完整模型前传: loss={loss.item():.4f}")


def test_full_model_with_pretrained():
    """加载预训练 TinyUSFM 权重后的前传测试。"""
    pretrained = "/home/lzj813/TinySpatial_Project/TinyUSFM.pth"
    if not os.path.exists(pretrained):
        print("[SKIP] 未找到预训练权重")
        return
    model = ACMMIMPretrainModel(pretrained_path=pretrained)
    bus = torch.randn(2, 1, 224, 224)
    swe = torch.randn(2, 3, 224, 224)
    loss, pred, mask = model(bus, swe)
    assert loss.dim() == 0
    print(f"[PASS] 预训练模型前传: loss={loss.item():.4f}")


def test_gradient_flow():
    """验证梯度可正常回传至编码器参数。"""
    model = ACMMIMPretrainModel(pretrained_path=None)
    bus = torch.randn(2, 1, 224, 224)
    swe = torch.randn(2, 3, 224, 224)
    loss, _, _ = model(bus, swe)
    loss.backward()

    # 检查编码器 block 0 是否有梯度
    has_grad = False
    for name, param in model.encoder.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "编码器无梯度回传！"
    print("[PASS] 梯度回传验证通过")


if __name__ == "__main__":
    test_joint_patch_embedding()
    test_masking_engine()
    test_decoder()
    test_reconstruction_loss()
    test_full_model_forward()
    test_full_model_with_pretrained()
    test_gradient_flow()
    print("\n所有 Stage 1 测试通过！")
