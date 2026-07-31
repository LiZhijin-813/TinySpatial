"""验证 Stage 2 分类头、类别权重与损失语义的契约。"""

import torch
import torch.nn as nn

from code.models.stage2.task_heads import Stage2TaskHeads
from code.train.stage2_objectives import (
    compute_class_weights,
    compute_stage2_losses,
)


def test_task_heads_expose_expected_logits():
    """平坦四类、平坦五类与双头模式分别输出约定形状的 logits。"""
    features = torch.randn(3, 16)

    assert Stage2TaskHeads(16, "flat4")(features)["class_logits"].shape == (3, 4)
    assert Stage2TaskHeads(16, "flat5")(features)["class_logits"].shape == (3, 5)

    dual = Stage2TaskHeads(16, "dual_head")(features)
    assert dual["malignancy_logits"].shape == (3, 2)
    assert dual["subtype_logits"].shape == (3, 4)


def test_dual_head_subtype_loss_ignores_benign():
    """双头亚型损失只纳入恶性样本，并以默认系数组合总损失。"""
    outputs = {
        "malignancy_logits": torch.tensor(
            [[3.0, 0.0], [0.0, 3.0], [0.0, 3.0]], requires_grad=True
        ),
        "subtype_logits": torch.tensor(
            [
                [9.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
            ],
            requires_grad=True,
        ),
    }
    batch = {
        "malignancy_label": torch.tensor([0, 1, 1]),
        "subtype_label": torch.tensor([-1, 0, 1]),
    }
    criteria = {
        "malignancy": nn.CrossEntropyLoss(),
        "subtype": nn.CrossEntropyLoss(),
    }

    losses = compute_stage2_losses(outputs, batch, "dual_head", criteria)
    expected_subtype = criteria["subtype"](
        outputs["subtype_logits"][1:], torch.tensor([0, 1])
    )

    assert torch.allclose(losses["subtype_loss"], expected_subtype)
    assert torch.allclose(
        losses["total_loss"],
        losses["subtype_loss"] + 0.3 * losses["malignancy_loss"],
    )


def test_benign_only_batch_has_finite_zero_subtype_loss():
    """全良性批次的亚型损失为可反传的有限零值。"""
    outputs = {
        "malignancy_logits": torch.randn(2, 2, requires_grad=True),
        "subtype_logits": torch.randn(2, 4, requires_grad=True),
    }
    batch = {
        "malignancy_label": torch.zeros(2, dtype=torch.long),
        "subtype_label": torch.full((2,), -1, dtype=torch.long),
    }
    criteria = {
        "malignancy": nn.CrossEntropyLoss(),
        "subtype": nn.CrossEntropyLoss(),
    }

    losses = compute_stage2_losses(outputs, batch, "dual_head", criteria)

    assert losses["subtype_loss"].item() == 0.0
    assert torch.isfinite(losses["total_loss"])

    losses["total_loss"].backward()
    assert outputs["subtype_logits"].grad is not None


def test_separate_class_weights_have_expected_shapes():
    """良恶性与亚型权重独立计算，忽略良性的无效亚型标签。"""
    samples = [
        {"malignancy_label": 0, "subtype_label": -1},
        {"malignancy_label": 1, "subtype_label": 0},
        {"malignancy_label": 1, "subtype_label": 1},
        {"malignancy_label": 1, "subtype_label": 1},
    ]

    binary = compute_class_weights(samples, "malignancy_label", 2)
    subtype = compute_class_weights(samples, "subtype_label", 4, ignore_index=-1)

    assert binary.shape == (2,)
    assert subtype.shape == (4,)
    assert torch.isfinite(binary).all()
    assert torch.isfinite(subtype).all()
