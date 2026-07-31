"""结构化 Stage 2 推理接口的契约测试。"""

from types import SimpleNamespace

import pytest
import torch

import code.train.inference as inference_module


class FakeStructuredModel(torch.nn.Module):
    """返回固定结构化输出的轻量模型替身。"""

    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def forward(self, bus_img, swe_img, cdfi_img, input_ids, attention_mask):
        return self.outputs


class RecordingModel:
    """记录检查点加载参数的最小模型替身。"""

    def __init__(self):
        self.loaded_state_dict = None
        self.strict = None

    def load_state_dict(self, state_dict, strict):
        self.loaded_state_dict = state_dict
        self.strict = strict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


@pytest.mark.parametrize(
    ("task_mode", "outputs", "expected_predictions"),
    [
        (
            "flat4",
            {"class_logits": torch.tensor([[0.0, 3.0, 1.0, 2.0]])},
            torch.tensor([1]),
        ),
        (
            "flat5",
            {"class_logits": torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])},
            torch.tensor([4]),
        ),
    ],
)
def test_flat_task_predictions_and_probabilities(task_mode, outputs, expected_predictions):
    """平坦任务应对 class_logits 计算 softmax 与最大概率类别。"""
    predictions, probabilities = inference_module.task_predictions_and_probabilities(
        outputs,
        task_mode,
    )

    assert torch.equal(predictions, expected_predictions)
    assert torch.allclose(
        probabilities,
        torch.softmax(outputs["class_logits"], dim=1),
    )


def test_dual_head_predictions_and_probabilities_follow_hard_rule():
    """双头任务按良恶性硬规则预测，并组合为五类概率。"""
    outputs = {
        "malignancy_logits": torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
        "subtype_logits": torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 2.0]]
        ),
    }

    predictions, probabilities = inference_module.task_predictions_and_probabilities(
        outputs,
        "dual_head",
    )

    malignancy_probabilities = torch.softmax(outputs["malignancy_logits"], dim=1)
    subtype_probabilities = torch.softmax(outputs["subtype_logits"], dim=1)
    expected_probabilities = torch.cat(
        [
            malignancy_probabilities[:, 1:2] * subtype_probabilities,
            malignancy_probabilities[:, 0:1],
        ],
        dim=1,
    )

    assert torch.equal(predictions, torch.tensor([4, 2]))
    assert torch.allclose(probabilities, expected_probabilities)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))


@pytest.mark.parametrize(
    ("task_mode", "outputs", "expected_labels"),
    [
        (
            "flat4",
            {"class_logits": torch.tensor([[0.0, 3.0, 1.0, 2.0]])},
            [1],
        ),
        (
            "flat5",
            {"class_logits": torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])},
            [4],
        ),
        (
            "dual_head",
            {
                "malignancy_logits": torch.tensor([[0.0, 5.0]]),
                "subtype_logits": torch.tensor([[0.0, 1.0, 4.0, 2.0]]),
            },
            [4],
        ),
    ],
)
def test_predict_batch_selects_labels_for_task_mode(
    task_mode,
    outputs,
    expected_labels,
):
    """批量推理应按任务模式选择对应的真实标签。"""
    batch = {
        "bus_img": torch.zeros(1, 1, 1, 1),
        "swe_img": torch.zeros(1, 3, 1, 1),
        "cdfi_img": torch.zeros(1, 3, 1, 1),
        "input_ids": torch.zeros(1, 1, dtype=torch.long),
        "attention_mask": torch.ones(1, 1, dtype=torch.long),
        "subtype_label": torch.tensor([1]),
        "class_label": torch.tensor([4]),
    }

    _, labels, _ = inference_module.predict_batch(
        FakeStructuredModel(outputs),
        [batch],
        torch.device("cpu"),
        task_mode,
    )

    assert labels.tolist() == expected_labels


@pytest.mark.parametrize(
    ("state_dict", "error_message"),
    [
        ({"encoder.norm.weight": torch.ones(1)}, "任务头"),
        (
            {
                "task_heads.class_head.1.weight": torch.ones(1),
                "mlp_head.1.weight": torch.ones(1),
            },
            "旧版.*mlp_head",
        ),
    ],
)
def test_checkpoint_loader_rejects_incompatible_classifier_heads(
    tmp_path,
    capsys,
    state_dict,
    error_message,
):
    """检查点加载器必须拒绝缺失任务头或残留旧分类头的检查点。"""
    checkpoint_path = tmp_path / "legacy_checkpoint.pth"
    torch.save({"model_state_dict": state_dict}, checkpoint_path)
    model = RecordingModel()

    with pytest.raises(ValueError, match=error_message):
        inference_module.load_stage2_checkpoint(
            model,
            checkpoint_path,
            torch.device("cpu"),
        )

    assert model.loaded_state_dict is None
    assert capsys.readouterr().out == ""


def test_checkpoint_loader_allows_noncritical_differences(tmp_path):
    """检查点加载器应以非严格模式保留兼容任务头并忽略非关键差异。"""
    checkpoint_path = tmp_path / "compatible_checkpoint.pth"
    state_dict = {
        "task_heads.class_head.1.weight": torch.ones(1),
        "obsolete_auxiliary.weight": torch.ones(1),
    }
    torch.save({"model": state_dict}, checkpoint_path)
    model = RecordingModel()

    inference_module.load_stage2_checkpoint(
        model,
        checkpoint_path,
        torch.device("cpu"),
    )

    assert model.strict is False
    assert model.loaded_state_dict == state_dict
