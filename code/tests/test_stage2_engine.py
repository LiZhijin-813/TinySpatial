"""定义 Stage 2 单轮训练与任务感知评估引擎契约。"""

import math
import re

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from code.train.stage2_engine import (
    evaluate_loader,
    monitor_value,
    move_batch_to_device,
    train_one_epoch,
)


class _NonTensorWithTo:
    """带有 to 方法但不是张量的元数据哨兵。"""

    def to(self, device):
        """若引擎错误移动非张量对象则立即失败。"""
        raise AssertionError(f"不应移动非张量对象到 {device}")


class _RecordingCriterion:
    """记录目标标签并计算可反传交叉熵的轻量损失。"""

    def __init__(self):
        self.targets = []

    def __call__(self, logits, targets):
        self.targets.append(targets.detach().cpu().clone())
        return F.cross_entropy(logits, targets)


class _UnitGradientCriterion:
    """为每个 batch 产生单位梯度以检查累积缩放。"""

    def __call__(self, logits, targets):
        return logits[:, 0].mean()


class _OverfitTrainingModel(nn.Module):
    """只接收 BUS 输入的可微分四分类假模型。"""

    def __init__(self):
        super().__init__()
        self.class_bias = nn.Parameter(torch.zeros(4))

    def forward(self, bus_img):
        batch_size = bus_img.shape[0]
        return {
            "class_logits": self.class_bias.unsqueeze(0).expand(batch_size, -1),
            "image_features": bus_img,
        }


class _MultimodalTrainingModel(nn.Module):
    """同时提供平坦头和双头输出的可微分假模型。"""

    def __init__(self):
        super().__init__()
        self.class_bias = nn.Parameter(torch.zeros(5))
        self.malignancy_bias = nn.Parameter(torch.zeros(2))
        self.subtype_bias = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        bus_img,
        swe_img,
        cdfi_img,
        input_ids,
        attention_mask,
    ):
        batch_size = bus_img.shape[0]
        return {
            "class_logits": self.class_bias.unsqueeze(0).expand(batch_size, -1),
            "malignancy_logits": self.malignancy_bias.unsqueeze(0).expand(
                batch_size, -1
            ),
            "subtype_logits": self.subtype_bias.unsqueeze(0).expand(
                batch_size, -1
            ),
            "image_features": bus_img,
        }


class _EvaluationModel(nn.Module):
    """将内存 batch 中的张量直接解释为评估 logits。"""

    def __init__(self, dual_head=False):
        super().__init__()
        self.dual_head = dual_head

    def forward(
        self,
        bus_img,
        swe_img,
        cdfi_img,
        input_ids,
        attention_mask,
    ):
        if self.dual_head:
            return {
                "malignancy_logits": bus_img,
                "subtype_logits": swe_img,
                "image_features": cdfi_img,
            }
        return {
            "class_logits": bus_img,
            "image_features": cdfi_img,
        }


class _OverfitEvaluationModel(nn.Module):
    """将 BUS 输入直接解释为过拟合探针 logits 的假模型。"""

    def forward(self, bus_img):
        return {
            "class_logits": bus_img,
            "image_features": bus_img[:, :2],
        }


class _CountingSGD(torch.optim.SGD):
    """记录真实 optimizer.step 调用次数的 SGD。"""

    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class _DeceptiveLoader:
    """长度报告与真实迭代批次数不同的数据加载器。"""

    def __init__(self, batches, reported_len):
        self.batches = batches
        self.reported_len = reported_len

    def __len__(self):
        return self.reported_len

    def __iter__(self):
        return iter(self.batches)


def _training_batch(subtype_labels=(2, 1)):
    """构造不依赖图像、文本模型或 GPU 的内存训练 batch。"""
    batch_size = len(subtype_labels)
    return {
        "bus_img": torch.ones(batch_size, 1),
        "swe_img": torch.ones(batch_size, 1),
        "cdfi_img": torch.ones(batch_size, 1),
        "input_ids": torch.ones(batch_size, 2, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 2, dtype=torch.long),
        "class_label": torch.tensor([4, 1], dtype=torch.long)[:batch_size],
        "subtype_label": torch.tensor(subtype_labels, dtype=torch.long),
        "malignancy_label": torch.tensor([0, 1], dtype=torch.long)[:batch_size],
        "case_id": [f"case-{index}" for index in range(batch_size)],
    }


def _evaluation_batch(
    class_or_binary_logits,
    *,
    subtype_logits=None,
    subtype_labels=None,
    malignancy_labels=None,
    image_features=None,
):
    """构造将输入直接作为 logits 的内存评估 batch。"""
    batch_size = class_or_binary_logits.shape[0]
    if subtype_logits is None:
        subtype_logits = torch.zeros(batch_size, 4)
    if image_features is None:
        image_features = torch.arange(
            batch_size * 2, dtype=torch.float32
        ).reshape(batch_size, 2)
    batch = {
        "bus_img": class_or_binary_logits,
        "swe_img": subtype_logits,
        "cdfi_img": image_features,
        "input_ids": torch.zeros(batch_size, 1, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 1, dtype=torch.long),
        "case_id": [f"case-{index}" for index in range(batch_size)],
    }
    if subtype_labels is not None:
        batch["subtype_label"] = torch.tensor(subtype_labels, dtype=torch.long)
    if malignancy_labels is not None:
        batch["malignancy_label"] = torch.tensor(
            malignancy_labels, dtype=torch.long
        )
    return batch


def _assert_chinese_value_error(error):
    """确认异常类型之外还提供可读中文错误信息。"""
    assert re.search(r"[\u4e00-\u9fff]", str(error.value))


def test_move_batch_to_device_moves_only_tensors():
    """批次迁移只处理张量并原样保留非张量元数据。"""
    marker = _NonTensorWithTo()
    original = {
        "tensor": torch.tensor([1.0]),
        "case_id": ["001"],
        "metadata": marker,
    }

    moved = move_batch_to_device(original, torch.device("cpu"))

    assert moved is not original
    assert moved["tensor"].device.type == "cpu"
    assert moved["case_id"] is original["case_id"]
    assert moved["metadata"] is marker


@pytest.mark.parametrize(
    "task_mode, label_key, expected_loss_keys",
    [
        ("overfit", "subtype_label", {"total_loss"}),
        ("flat4", "subtype_label", {"total_loss", "class_loss"}),
        ("flat5", "class_label", {"total_loss", "class_loss"}),
    ],
)
def test_flat_training_modes_use_expected_labels_and_loss_paths(
    task_mode,
    label_key,
    expected_loss_keys,
):
    """overfit、flat4 与 flat5 必须选择各自约定的标签和损失路径。"""
    batch = _training_batch()
    criterion = _RecordingCriterion()
    model = (
        _OverfitTrainingModel()
        if task_mode == "overfit"
        else _MultimodalTrainingModel()
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = train_one_epoch(
        model,
        [batch],
        optimizer,
        {"class": criterion},
        task_mode,
        torch.device("cpu"),
        max_grad_norm=0,
    )

    assert len(criterion.targets) == 1
    assert torch.equal(criterion.targets[0], batch[label_key])
    assert expected_loss_keys <= metrics.keys()
    assert math.isfinite(metrics["total_loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_dual_head_training_uses_binary_and_malignant_subtype_losses():
    """双头训练必须对全体做良恶性损失且只对恶性样本做亚型损失。"""
    batch = _training_batch(subtype_labels=(-1, 1))
    malignancy_criterion = _RecordingCriterion()
    subtype_criterion = _RecordingCriterion()
    model = _MultimodalTrainingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = train_one_epoch(
        model,
        [batch],
        optimizer,
        {
            "malignancy": malignancy_criterion,
            "subtype": subtype_criterion,
        },
        "dual_head",
        torch.device("cpu"),
        lambda_bm=0.3,
        max_grad_norm=0,
    )

    assert torch.equal(
        malignancy_criterion.targets[0],
        batch["malignancy_label"],
    )
    assert torch.equal(subtype_criterion.targets[0], torch.tensor([1]))
    assert {
        "total_loss",
        "malignancy_loss",
        "subtype_loss",
        "accuracy",
    } <= metrics.keys()


@pytest.mark.parametrize("reported_len", [4, 2])
def test_gradient_accumulation_uses_actual_batches_not_reported_length(
    reported_len,
):
    """梯度累积必须按真实迭代分组而不是依赖 loader 报告长度。"""
    model = _OverfitTrainingModel()
    optimizer = _CountingSGD(model.parameters(), lr=1.0)
    batches = [
        _training_batch(subtype_labels=(0,)),
        _training_batch(subtype_labels=(0,)),
        _training_batch(subtype_labels=(0,)),
    ]
    loader = _DeceptiveLoader(batches, reported_len)

    train_one_epoch(
        model,
        loader,
        optimizer,
        {"class": _UnitGradientCriterion()},
        "overfit",
        torch.device("cpu"),
        max_grad_norm=0,
        accum_steps=2,
    )

    assert optimizer.step_count == 2
    assert model.class_bias[0].item() == pytest.approx(-2.0)


def test_flat4_training_accuracy_uses_subtype_labels():
    """flat4 accuracy 必须按 subtype_label 统计而不是 class_label。"""
    batch = _training_batch(subtype_labels=(0, 0))
    batch["class_label"] = torch.tensor([4, 1])
    model = _MultimodalTrainingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = train_one_epoch(
        model,
        [batch],
        optimizer,
        {"class": _RecordingCriterion()},
        "flat4",
        torch.device("cpu"),
        max_grad_norm=0,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)


def test_flat5_training_accuracy_uses_class_labels():
    """flat5 accuracy 必须按 class_label 统计而不是 subtype_label。"""
    batch = _training_batch(subtype_labels=(1, 1))
    batch["class_label"] = torch.tensor([0, 1])
    model = _MultimodalTrainingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = train_one_epoch(
        model,
        [batch],
        optimizer,
        {"class": _RecordingCriterion()},
        "flat5",
        torch.device("cpu"),
        max_grad_norm=0,
    )

    assert metrics["accuracy"] == pytest.approx(0.5)


def test_dual_training_accuracy_uses_only_malignant_subtypes():
    """dual_head accuracy 分母必须排除良性样本。"""
    batch = _training_batch(subtype_labels=(1, 0))
    model = _MultimodalTrainingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = train_one_epoch(
        model,
        [batch],
        optimizer,
        {
            "malignancy": _RecordingCriterion(),
            "subtype": _RecordingCriterion(),
        },
        "dual_head",
        torch.device("cpu"),
        max_grad_norm=0,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)


@pytest.mark.parametrize("accum_steps", [0, -1, 1.5, True])
def test_train_rejects_non_positive_integer_accum_steps(accum_steps):
    """accum_steps 必须是非布尔正整数。"""
    model = _OverfitTrainingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError) as error:
        train_one_epoch(
            model,
            [_training_batch()],
            optimizer,
            {"class": _RecordingCriterion()},
            "overfit",
            torch.device("cpu"),
            accum_steps=accum_steps,
        )

    _assert_chinese_value_error(error)


def test_train_rejects_empty_loader():
    """训练数据加载器为空时必须以中文 ValueError 明确失败。"""
    model = _OverfitTrainingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError) as error:
        train_one_epoch(
            model,
            [],
            optimizer,
            {"class": _RecordingCriterion()},
            "overfit",
            torch.device("cpu"),
        )

    _assert_chinese_value_error(error)


def test_train_rejects_unknown_task_mode():
    """训练阶段必须拒绝未知任务模式并给出中文错误。"""
    model = _MultimodalTrainingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError) as error:
        train_one_epoch(
            model,
            [_training_batch()],
            optimizer,
            {"class": _RecordingCriterion()},
            "unknown",
            torch.device("cpu"),
        )

    _assert_chinese_value_error(error)


def test_evaluate_overfit_reports_malignant_metrics():
    """overfit 评估必须通过 BUS-only 调用返回四亚型恶性指标。"""
    batch = _evaluation_batch(
        torch.eye(4) * 5,
        subtype_labels=[0, 1, 2, 3],
    )

    metrics = evaluate_loader(
        _OverfitEvaluationModel(),
        [batch],
        "overfit",
        torch.device("cpu"),
        "malignant",
    )

    assert metrics["malignant"]["accuracy"] == pytest.approx(1.0)
    assert metrics["malignant"]["macro_f1"] == pytest.approx(1.0)
    assert "diagnostic" in metrics


def test_evaluate_flat4_reports_malignant_metrics():
    """flat4 评估必须返回四亚型恶性指标。"""
    batch = _evaluation_batch(
        torch.eye(4) * 5,
        subtype_labels=[0, 1, 2, 3],
    )

    metrics = evaluate_loader(
        _EvaluationModel(),
        [batch],
        "flat4",
        torch.device("cpu"),
        "malignant",
    )

    assert metrics["malignant"]["accuracy"] == pytest.approx(1.0)
    assert metrics["malignant"]["macro_f1"] == pytest.approx(1.0)
    assert "diagnostic" in metrics


def test_evaluate_flat5_malignant_reports_conditional_and_end_to_end_metrics():
    """flat5 恶性集必须同时报告条件亚型与端到端五分类语义。"""
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0, 6.0],
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, 0.0],
        ]
    )
    batch = _evaluation_batch(logits, subtype_labels=[0, 1, 2, 3])

    metrics = evaluate_loader(
        _EvaluationModel(),
        [batch],
        "flat5",
        torch.device("cpu"),
        "malignant",
    )

    assert metrics["malignant"]["macro_f1"] == pytest.approx(1.0)
    assert metrics["malignant_end_to_end"]["accuracy"] == pytest.approx(0.75)
    assert metrics["malignant_end_to_end"]["macro_f1"] == pytest.approx(0.75)
    assert metrics["malignant_end_to_end"][
        "balanced_accuracy"
    ] == pytest.approx(0.75)
    assert "overall" not in metrics
    assert monitor_value(metrics, "malignant_macro_f1") == pytest.approx(1.0)


def test_evaluate_flat5_binary_reports_binary_metrics():
    """flat5 良恶性集必须将类别四解释为良性并报告二分类指标。"""
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.0],
            [0.0, 4.0, 0.0, 0.0, 0.0],
        ]
    )
    batch = _evaluation_batch(
        logits,
        malignancy_labels=[0, 1, 0, 1],
    )

    metrics = evaluate_loader(
        _EvaluationModel(),
        [batch],
        "flat5",
        torch.device("cpu"),
        "binary",
    )

    assert metrics["binary"]["accuracy"] == pytest.approx(1.0)
    assert metrics["binary"]["auc"] == pytest.approx(1.0)


def test_evaluate_dual_malignant_reports_conditional_and_end_to_end_metrics():
    """dual_head 恶性集必须分别评估条件亚型与良恶性门控后的端到端结果。"""
    malignancy_logits = torch.tensor(
        [
            [5.0, 0.0],
            [0.0, 5.0],
            [0.0, 5.0],
            [0.0, 5.0],
        ]
    )
    subtype_logits = torch.eye(4) * 5
    batch = _evaluation_batch(
        malignancy_logits,
        subtype_logits=subtype_logits,
        subtype_labels=[0, 1, 2, 3],
    )

    metrics = evaluate_loader(
        _EvaluationModel(dual_head=True),
        [batch],
        "dual_head",
        torch.device("cpu"),
        "malignant",
    )

    assert metrics["malignant"]["macro_f1"] == pytest.approx(1.0)
    assert metrics["malignant_end_to_end"]["accuracy"] == pytest.approx(0.75)
    assert metrics["malignant_end_to_end"]["macro_f1"] == pytest.approx(0.75)
    assert metrics["malignant_end_to_end"][
        "balanced_accuracy"
    ] == pytest.approx(0.75)


def test_evaluate_dual_binary_reports_binary_metrics():
    """dual_head 良恶性集必须使用二分类头概率与预测。"""
    malignancy_logits = torch.tensor(
        [
            [5.0, 0.0],
            [0.0, 5.0],
            [4.0, 0.0],
            [0.0, 4.0],
        ]
    )
    batch = _evaluation_batch(
        malignancy_logits,
        subtype_logits=torch.zeros(4, 4),
        malignancy_labels=[0, 1, 0, 1],
    )

    metrics = evaluate_loader(
        _EvaluationModel(dual_head=True),
        [batch],
        "dual_head",
        torch.device("cpu"),
        "binary",
    )

    assert metrics["binary"]["accuracy"] == pytest.approx(1.0)
    assert metrics["binary"]["auc"] == pytest.approx(1.0)


def test_single_sample_diagnostics_use_finite_population_variance():
    """单样本 logits 与特征方差必须使用 population 语义并返回有限零值。"""
    batch = _evaluation_batch(
        torch.tensor([[3.0, 1.0, 0.0, -1.0]]),
        subtype_labels=[0],
        image_features=torch.tensor([[7.0, 9.0]]),
    )

    metrics = evaluate_loader(
        _EvaluationModel(),
        [batch],
        "flat4",
        torch.device("cpu"),
        "malignant",
    )

    assert math.isfinite(metrics["diagnostic"]["logits_var"])
    assert math.isfinite(metrics["diagnostic"]["feature_var"])
    assert metrics["diagnostic"]["logits_var"] == pytest.approx(0.0)
    assert metrics["diagnostic"]["feature_var"] == pytest.approx(0.0)


def test_evaluate_rejects_unknown_task_mode():
    """评估阶段必须拒绝未知任务模式并给出中文错误。"""
    batch = _evaluation_batch(
        torch.eye(4),
        subtype_labels=[0, 1, 2, 3],
    )

    with pytest.raises(ValueError) as error:
        evaluate_loader(
            _EvaluationModel(),
            [batch],
            "unknown",
            torch.device("cpu"),
            "malignant",
        )

    _assert_chinese_value_error(error)


def test_evaluate_rejects_unknown_split_kind():
    """评估阶段必须拒绝未知数据集语义并给出中文错误。"""
    batch = _evaluation_batch(
        torch.eye(4),
        subtype_labels=[0, 1, 2, 3],
    )

    with pytest.raises(ValueError) as error:
        evaluate_loader(
            _EvaluationModel(),
            [batch],
            "flat4",
            torch.device("cpu"),
            "unknown",
        )

    _assert_chinese_value_error(error)


def test_evaluate_rejects_empty_loader():
    """评估数据加载器为空时必须以中文 ValueError 明确失败。"""
    with pytest.raises(ValueError) as error:
        evaluate_loader(
            _EvaluationModel(),
            [],
            "flat4",
            torch.device("cpu"),
            "malignant",
        )

    _assert_chinese_value_error(error)
