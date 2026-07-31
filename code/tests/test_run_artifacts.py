"""定义 Stage 2 JSON 运行产物与模型选择指标契约。"""

import json
import re
from argparse import Namespace

import numpy as np
import pytest
import torch

from code.train.run_artifacts import (
    initialize_run_artifacts,
    save_json,
    save_training_state,
)
from code.train.stage2_engine import monitor_value


def _assert_chinese_value_error(error):
    """确认异常类型之外还提供可读中文错误信息。"""
    assert re.search(r"[\u4e00-\u9fff]", str(error.value))


def test_run_artifacts_write_required_json_files(tmp_path):
    """初始化与训练状态保存必须生成四个约定的 JSON 文件。"""
    args = Namespace(task_mode="dual_head", seed=42)
    manifest = {"task_mode": "dual_head", "splits": {"train": ["001"]}}

    initialize_run_artifacts(tmp_path, args, manifest)
    save_training_state(
        tmp_path,
        history=[{"epoch": 1, "train": {"total_loss": 1.0}}],
        best_metrics={"malignant_macro_f1": 0.4},
    )

    assert json.loads(
        (tmp_path / "args.json").read_text(encoding="utf-8")
    ) == vars(args)
    assert json.loads(
        (tmp_path / "split_manifest.json").read_text(encoding="utf-8")
    ) == manifest
    assert json.loads(
        (tmp_path / "history.json").read_text(encoding="utf-8")
    ) == [{"epoch": 1, "train": {"total_loss": 1.0}}]
    assert json.loads(
        (tmp_path / "metrics_best.json").read_text(encoding="utf-8")
    ) == {"malignant_macro_f1": 0.4}


def test_json_artifacts_serialize_namespace_numpy_and_torch_scalars(tmp_path):
    """Namespace、NumPy 标量与 Torch 标量必须可写入标准 JSON。"""
    args = Namespace(
        seed=np.int64(7),
        threshold=torch.tensor(0.25),
    )
    manifest = {
        "fold": np.int32(2),
        "score": np.float32(0.5),
        "count": torch.tensor(3),
    }

    initialize_run_artifacts(tmp_path, args, manifest)

    saved_args = json.loads((tmp_path / "args.json").read_text(encoding="utf-8"))
    saved_manifest = json.loads(
        (tmp_path / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_args == {"seed": 7, "threshold": pytest.approx(0.25)}
    assert saved_manifest == {
        "fold": 2,
        "score": pytest.approx(0.5),
        "count": 3,
    }


def test_save_json_preserves_utf8_chinese(tmp_path):
    """JSON 文件必须保留中文原文而不是写成 Unicode 转义。"""
    path = tmp_path / "中文产物.json"

    save_json(path, {"病例": "恶性亚型", "说明": "可审计训练"})

    raw_text = path.read_text(encoding="utf-8")
    assert "病例" in raw_text
    assert "恶性亚型" in raw_text
    assert "\\u75c5\\u4f8b" not in raw_text
    assert json.loads(raw_text)["说明"] == "可审计训练"


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float32(np.nan),
        np.array([1.0, np.inf]),
        torch.tensor(float("nan")),
        torch.tensor([1.0, float("inf")]),
    ],
)
def test_save_json_rejects_non_finite_numbers(tmp_path, value):
    """严格 JSON 必须拒绝 Python、NumPy 与 Torch 中的 NaN 和 Inf。"""
    path = tmp_path / "invalid.json"

    with pytest.raises(ValueError) as error:
        save_json(path, {"value": value})

    _assert_chinese_value_error(error)
    assert not path.exists()


def test_monitor_metric_contract_uses_named_sections():
    """显式诊断模式下兼容监控项必须从约定指标分区读取数值。"""
    metrics = {
        "malignant": {
            "macro_f1": 0.41,
            "balanced_accuracy": 0.43,
            "accuracy": 0.45,
        },
        "overall": {
            "macro_f1": 0.81,
            "balanced_accuracy": 0.82,
            "accuracy": 0.72,
        },
        "binary": {"macro_f1": 0.91, "accuracy": 0.92},
    }

    assert monitor_value(metrics, "malignant_macro_f1") == pytest.approx(0.41)
    assert monitor_value(
        metrics,
        "macro_f1",
        allow_diagnostic=True,
    ) == pytest.approx(0.41)
    assert monitor_value(
        metrics,
        "balanced_acc",
        allow_diagnostic=True,
    ) == pytest.approx(0.43)
    assert monitor_value(
        metrics,
        "acc",
        allow_diagnostic=True,
    ) == pytest.approx(0.72)


@pytest.mark.parametrize("monitor_metric", ["macro_f1", "balanced_acc", "acc"])
def test_diagnostic_monitor_metrics_require_explicit_opt_in(monitor_metric):
    """默认模型选择必须拒绝所有诊断监控项。"""
    metrics = {
        "malignant": {
            "macro_f1": 0.41,
            "balanced_accuracy": 0.43,
            "accuracy": 0.45,
        },
        "overall": {"accuracy": 0.72},
    }

    with pytest.raises(ValueError) as error:
        monitor_value(metrics, monitor_metric)

    _assert_chinese_value_error(error)


def test_malignant_macro_f1_reads_only_malignant_macro_f1():
    """主模型选择值不得读取 flat5 总体准确率或其他宏平均指标。"""
    metrics = {
        "malignant": {"macro_f1": np.float32(0.37)},
        "malignant_end_to_end": {"macro_f1": 0.88, "accuracy": 0.99},
        "overall": {"macro_f1": 0.97, "accuracy": 1.0},
        "binary": {"macro_f1": 0.96, "accuracy": 1.0},
    }

    assert monitor_value(metrics, "malignant_macro_f1") == pytest.approx(0.37)


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        [],
        {},
        {"malignant": None},
        {"malignant": {}},
        {"malignant": {"accuracy": 0.9}},
    ],
)
def test_primary_monitor_rejects_missing_metrics_sections_and_keys(metrics):
    """主监控必须以中文 ValueError 拒绝缺失、空或非法指标结构。"""
    with pytest.raises(ValueError) as error:
        monitor_value(metrics, "malignant_macro_f1")

    _assert_chinese_value_error(error)


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        "0.4",
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_primary_monitor_rejects_bool_non_numeric_and_non_finite_values(value):
    """主监控值必须是非布尔且有限的实数。"""
    metrics = {"malignant": {"macro_f1": value}}

    with pytest.raises(ValueError) as error:
        monitor_value(metrics, "malignant_macro_f1")

    _assert_chinese_value_error(error)


@pytest.mark.parametrize(
    "monitor_metric, metrics",
    [
        ("macro_f1", {"malignant": {"accuracy": 0.8}}),
        ("balanced_acc", {"malignant": {"macro_f1": 0.8}}),
        ("acc", {"overall": {}}),
    ],
)
def test_diagnostic_monitor_rejects_missing_metric_keys(
    monitor_metric,
    metrics,
):
    """显式诊断模式也必须以中文 ValueError 拒绝缺失指标键。"""
    with pytest.raises(ValueError) as error:
        monitor_value(
            metrics,
            monitor_metric,
            allow_diagnostic=True,
        )

    _assert_chinese_value_error(error)


def test_unknown_monitor_metric_raises_chinese_value_error():
    """未知监控指标必须以中文 ValueError 明确拒绝。"""
    with pytest.raises(ValueError) as error:
        monitor_value({"malignant": {"macro_f1": 0.4}}, "flat5_accuracy")

    message = str(error.value)
    assert "monitor_metric" in message
    _assert_chinese_value_error(error)
