"""Stage 2 模块与完整模型契约测试。"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import code.models.stage2.subtyping_model as subtyping_module
import code.models.stage2.text_branch as text_branch_module
from code.models.stage2.cdfi_branch import CDFIBranch
from code.models.stage2.cross_attention import LightweightCrossAttention
from code.models.stage2.ip_adapter import DecoupledAttentionLayer


class FakeBert(nn.Module):
    """用于隔离外部预训练模型加载的 BERT 替身。"""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 768)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeTextBranch(nn.Module):
    """用于完整模型测试的轻量文本分支替身。"""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 512)

    def forward(self, input_ids, attention_mask):
        pooled = torch.nn.functional.one_hot(
            input_ids[:, 0] % 8,
            num_classes=8,
        ).float()
        return self.projection(pooled)


def _inputs(batch_size=2):
    """构造 32 像素输入，避免完整模型测试消耗额外资源。"""
    return (
        torch.randn(batch_size, 1, 32, 32),
        torch.randn(batch_size, 3, 32, 32),
        torch.randn(batch_size, 3, 32, 32),
        torch.randint(0, 100, (batch_size, 8)),
        torch.ones(batch_size, 8, dtype=torch.long),
    )


def _has_finite_nonzero_gradient(parameters):
    """确认至少一个参数获得有限且非零的梯度。"""
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in parameters
    )


def test_cdfi_branch():
    """CDFI 分支应输出与嵌入维度对齐的空间 token。"""
    model = CDFIBranch(out_dim=48, img_size=32)
    output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 1, 48)


def test_ip_adapter():
    """IP-Adapter 在缩放变化后仍保持 token 形状。"""
    layer = DecoupledAttentionLayer(embed_dim=48, num_heads=4, cdfi_dim=48)
    tokens = torch.randn(2, 5, 48)
    cdfi_tokens = torch.randn(2, 1, 48)

    assert layer(tokens, cdfi_tokens).shape == (2, 5, 48)

    layer.set_ip_adapter_scale(0.5)
    assert layer(tokens, cdfi_tokens).shape == (2, 5, 48)


def test_text_branch_without_network(monkeypatch):
    """文本分支在本地替身 BERT 下保持冻结与投影契约。"""
    monkeypatch.setattr(
        text_branch_module.AutoModel,
        "from_pretrained",
        lambda _name: FakeBert(),
    )
    model = text_branch_module.TextLogicBranch()

    output = model(
        torch.randint(0, 1000, (2, 8)),
        torch.ones(2, 8, dtype=torch.long),
    )

    assert output.shape == (2, 512)
    assert all(not parameter.requires_grad for parameter in model.bert.parameters())
    assert all(parameter.requires_grad for parameter in model.projection.parameters())


def test_cross_attention():
    """跨模态融合输出应保留图像特征并响应文本变化。"""
    model = LightweightCrossAttention(image_dim=48, text_dim=512)
    image_features = torch.randn(2, 48)
    text_features = torch.randn(2, 512)

    output = model(image_features, text_features)
    output_without_text = model(image_features, torch.zeros_like(text_features))

    assert output.shape == (2, 560)
    assert not torch.allclose(output, output_without_text, atol=1e-4)


@pytest.mark.parametrize(
    ("task_mode", "logit_shapes"),
    [
        ("flat4", {"class_logits": (2, 4)}),
        ("flat5", {"class_logits": (2, 5)}),
        (
            "dual_head",
            {
                "malignancy_logits": (2, 2),
                "subtype_logits": (2, 4),
            },
        ),
    ],
)
def test_full_model_returns_task_aware_dictionary(monkeypatch, task_mode, logit_shapes):
    """完整模型应按任务模式返回结构化输出。"""
    monkeypatch.setattr(subtyping_module, "TextLogicBranch", FakeTextBranch)
    model = subtyping_module.SECSubtypingModel(
        pretrained_path=None,
        task_mode=task_mode,
        img_size=32,
        patch_size=16,
        embed_dim=48,
        depth=2,
        num_heads=4,
        ip_adapter_layers=[1],
        unfreeze_last_n=1,
    )

    output = model(*_inputs())

    assert set(output) == {
        *logit_shapes,
        "image_features",
        "text_features",
        "fused_features",
    }
    assert output["image_features"].shape == (2, 48)
    assert output["text_features"].shape == (2, 512)
    assert output["fused_features"].shape == (2, 560)
    for name, shape in logit_shapes.items():
        assert output[name].shape == shape


def test_only_configured_encoder_layers_receive_gradients(monkeypatch):
    """冻结层无梯度，解冻层、任务头与适配器获得有效梯度。"""
    monkeypatch.setattr(subtyping_module, "TextLogicBranch", FakeTextBranch)
    model = subtyping_module.SECSubtypingModel(
        pretrained_path=None,
        task_mode="dual_head",
        img_size=32,
        patch_size=16,
        embed_dim=48,
        depth=2,
        num_heads=4,
        ip_adapter_layers=[1],
        unfreeze_last_n=1,
    )

    output = model(*_inputs())
    (output["malignancy_logits"].sum() + output["subtype_logits"].sum()).backward()

    frozen_parameters = tuple(model.encoder.blocks[0].parameters())
    assert all(not parameter.requires_grad for parameter in frozen_parameters)
    assert all(parameter.grad is None for parameter in frozen_parameters)
    assert _has_finite_nonzero_gradient(model.encoder.blocks[1].parameters())
    assert _has_finite_nonzero_gradient(model.task_heads.parameters())
    assert _has_finite_nonzero_gradient(model.ip_adapters.parameters())


def _small_model(monkeypatch, **overrides):
    """构造不加载外部文本模型的最小 Stage 2 模型。"""
    monkeypatch.setattr(subtyping_module, "TextLogicBranch", FakeTextBranch)
    settings = {
        "pretrained_path": None,
        "img_size": 32,
        "patch_size": 16,
        "embed_dim": 48,
        "depth": 2,
        "num_heads": 4,
        "ip_adapter_layers": [1],
        "unfreeze_last_n": 1,
    }
    settings.update(overrides)
    return subtyping_module.SECSubtypingModel(**settings)


def test_default_ip_adapter_layers_follow_encoder_depth(monkeypatch):
    """未指定注入层时，深度为二的编码器默认使用最后两层。"""
    model = _small_model(monkeypatch, ip_adapter_layers=None)

    assert model.ip_adapter_layers == [0, 1]
    assert set(model.ip_adapters) == {"0", "1"}


@pytest.mark.parametrize("invalid_unfreeze_last_n", [True, 1.5, -1, 3])
def test_unfreeze_last_n_requires_non_boolean_integer_in_range(
    monkeypatch,
    invalid_unfreeze_last_n,
):
    """解冻层数必须是范围内且非布尔值的整数。"""
    with pytest.raises(ValueError, match="unfreeze_last_n"):
        _small_model(monkeypatch, unfreeze_last_n=invalid_unfreeze_last_n)


def test_pretrained_loader_skips_mismatched_tensors_and_loads_matching_ones(
    monkeypatch,
    tmp_path,
):
    """预训练加载应忽略形状不符参数，同时载入形状匹配参数。"""
    model = _small_model(monkeypatch)
    expected_norm_weight = model.encoder.norm.weight.detach().clone() + 1
    checkpoint_path = tmp_path / "incompatible_checkpoint.pth"
    torch.save(
        {
            "model": {
                "pos_embed": torch.randn(
                    1,
                    model.encoder.pos_embed.shape[1] + 1,
                    model.encoder.pos_embed.shape[2],
                ),
                "norm.weight": expected_norm_weight,
            }
        },
        checkpoint_path,
    )

    model._load_pretrained(str(checkpoint_path))

    assert torch.equal(model.encoder.norm.weight, expected_norm_weight)
