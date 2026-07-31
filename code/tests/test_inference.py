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
    """记录检查点加载参数及其兼容性结果的最小模型替身。"""

    def __init__(self, missing_keys=None, unexpected_keys=None):
        self.loaded_state_dict = None
        self.strict = None
        self.missing_keys = missing_keys or []
        self.unexpected_keys = unexpected_keys or []

    def load_state_dict(self, state_dict, strict):
        self.loaded_state_dict = state_dict
        self.strict = strict
        return SimpleNamespace(
            missing_keys=self.missing_keys,
            unexpected_keys=self.unexpected_keys,
        )


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
    ("task_mode", "subtype_names", "class_logits", "expected_label"),
    [
        ("flat4", ["甲", "乙", "丙", "丁"], [0.0, 3.0, 1.0, 2.0], 1),
        (
            "flat5",
            ["甲", "乙", "丙", "丁", "良性"],
            [0.0, 3.0, 1.0, 2.0, 4.0],
            4,
        ),
    ],
)
def test_single_prediction_result_keeps_probabilities_for_flat_tasks(
    task_mode,
    subtype_names,
    class_logits,
    expected_label,
):
    """平坦任务的单例结果保留与预测标签一致的通用概率字段。"""
    outputs = {"class_logits": torch.tensor([class_logits])}
    predictions, probability_matrix = (
        inference_module.task_predictions_and_probabilities(outputs, task_mode)
    )

    result = inference_module.build_single_prediction_result(
        predictions,
        probability_matrix,
        outputs,
        task_mode,
        subtype_names,
    )

    assert result["predicted_label"] == expected_label
    assert result["predicted_subtype"] == subtype_names[expected_label]
    assert result["probabilities"] == pytest.approx(
        {
            name: probability_matrix[0, index].item()
            for index, name in enumerate(subtype_names)
        }
    )


def test_dual_head_single_result_separates_hard_decision_from_joint_probability():
    """双头单例结果必须显式区分硬门控决策与联合概率。"""
    subtype_names = ["甲", "乙", "丙", "丁", "良性"]
    outputs = {
        "malignancy_logits": torch.log(torch.tensor([[0.4, 0.6]])),
        "subtype_logits": torch.zeros(1, 4),
    }
    predictions, probability_matrix = (
        inference_module.task_predictions_and_probabilities(outputs, "dual_head")
    )

    result = inference_module.build_single_prediction_result(
        predictions,
        probability_matrix,
        outputs,
        "dual_head",
        subtype_names,
    )

    assert result["predicted_label"] == 0
    assert result["predicted_subtype"] == "甲"
    assert "硬门控" in result["decision_rule"]
    assert "probabilities" not in result
    assert result["malignancy_probabilities"] == pytest.approx(
        {"良性": 0.4, "恶性": 0.6}
    )
    assert result["conditional_subtype_probabilities"] == pytest.approx(
        {name: 0.25 for name in subtype_names[:4]}
    )
    assert result["joint_probabilities"] == pytest.approx(
        {
            "甲": 0.15,
            "乙": 0.15,
            "丙": 0.15,
            "丁": 0.15,
            "良性": 0.4,
        }
    )
    assert max(result["joint_probabilities"], key=result["joint_probabilities"].get) == "良性"


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


@pytest.mark.parametrize(
    "missing_key",
    [
        "encoder.norm.weight",
        "cdfi_branch.projection.0.weight",
        "cross_attention.text_proj.weight",
        "text_branch.projection.weight",
    ],
)
def test_checkpoint_loader_rejects_any_missing_production_parameter(
    tmp_path,
    capsys,
    missing_key,
):
    """检查点加载器必须拒绝任一生产参数缺失。"""
    checkpoint_path = tmp_path / "missing_parameter_checkpoint.pth"
    torch.save(
        {"model": {"task_heads.class_head.1.weight": torch.ones(1)}},
        checkpoint_path,
    )
    model = RecordingModel(missing_keys=[missing_key])

    with pytest.raises(ValueError, match="缺失"):
        inference_module.load_stage2_checkpoint(
            model,
            checkpoint_path,
            torch.device("cpu"),
        )

    assert capsys.readouterr().out == ""


def test_checkpoint_loader_rejects_any_unexpected_parameter(tmp_path, capsys):
    """检查点加载器必须拒绝多余辅助参数。"""
    checkpoint_path = tmp_path / "unexpected_parameter_checkpoint.pth"
    torch.save(
        {"model": {"task_heads.class_head.1.weight": torch.ones(1)}},
        checkpoint_path,
    )
    model = RecordingModel(unexpected_keys=["obsolete_auxiliary.weight"])

    with pytest.raises(ValueError, match="未预期"):
        inference_module.load_stage2_checkpoint(
            model,
            checkpoint_path,
            torch.device("cpu"),
        )

    assert capsys.readouterr().out == ""


def test_checkpoint_loader_allows_only_full_compatibility(tmp_path):
    """检查点加载器仅在完整兼容时以非严格模式完成加载。"""
    checkpoint_path = tmp_path / "compatible_checkpoint.pth"
    state_dict = {
        "task_heads.class_head.1.weight": torch.ones(1),
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


@pytest.mark.parametrize(
    ("task_mode", "expected_metadata_file"),
    [
        ("flat4", "metadata.csv"),
        ("flat5", "metadata_5class.csv"),
        ("dual_head", "metadata_5class.csv"),
    ],
)
def test_resolve_metadata_file_uses_task_mode_defaults(
    task_mode,
    expected_metadata_file,
):
    """未显式指定元数据文件时，应按任务模式选择默认文件。"""
    assert (
        inference_module.resolve_metadata_file(task_mode)
        == expected_metadata_file
    )


def test_resolve_metadata_file_preserves_explicit_value():
    """显式元数据文件应原样保留。"""
    assert (
        inference_module.resolve_metadata_file("flat5", "自定义.csv")
        == "自定义.csv"
    )


def test_resolve_metadata_file_rejects_unknown_task_mode():
    """未知任务模式必须给出中文错误。"""
    with pytest.raises(ValueError, match="任务模式"):
        inference_module.resolve_metadata_file("未知模式")
