"""层级探针的任务配置与标签筛选纯函数。"""

from __future__ import annotations

import math

import torch


def build_hierarchical_tasks():
    """构建层级标签探针所需的三组二分类任务配置。"""
    return {
        "luminal_vs_non_luminal": {
            "positive_labels": (0, 1),
            "negative_labels": (2, 3),
            "display_names": ("Luminal", "非Luminal"),
        },
        "luminal_a_vs_luminal_b": {
            "positive_labels": (0,),
            "negative_labels": (1,),
            "display_names": ("Luminal A", "Luminal B"),
        },
        "her2_vs_tnbc": {
            "positive_labels": (2,),
            "negative_labels": (3,),
            "display_names": ("HER2+", "TNBC"),
        },
    }


def select_hierarchical_labels(labels, task):
    """筛选任务相关标签，并映射为二元标签。"""
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("标签必须是一维张量")
    if labels.dtype not in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("标签必须是整数张量")
    if torch.any((labels < 0) | (labels >= 4)):
        raise ValueError("标签必须处于 0 到 3 的四分类空间内")

    positive_labels = tuple(task["positive_labels"])
    negative_labels = tuple(task["negative_labels"])

    positive = torch.zeros_like(labels, dtype=torch.bool)
    negative = torch.zeros_like(labels, dtype=torch.bool)
    for label in positive_labels:
        positive |= labels == label
    for label in negative_labels:
        negative |= labels == label

    mask = positive | negative
    binary_labels = torch.where(positive[mask], 0, 1)
    return mask, binary_labels


def aggregate_binary_metrics(fold_metrics):
    """汇总二分类任务在各折上的指标均值、标准差与逐折数值。"""
    if not fold_metrics:
        raise ValueError("折级指标不能为空")

    aggregated = {}
    for metric_name in ("macro_f1", "balanced_accuracy"):
        values = [float(metrics[metric_name]) for metrics in fold_metrics]
        mean_value = sum(values) / len(values)
        variance = sum(
            (value - mean_value) ** 2 for value in values
        ) / len(values)
        aggregated[metric_name] = {
            "mean": float(mean_value),
            "std": float(math.sqrt(variance)),
            "values": [float(value) for value in values],
        }
    return aggregated


def validate_fold_labels(train_labels, query_labels, task_name):
    """校验训练折与测试折标签是否满足二分类探针要求。"""

    def _validate_tensor(labels, split_name):
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
            raise ValueError(f"{task_name}的{split_name}标签必须是一维整数张量")
        if labels.dtype not in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"{task_name}的{split_name}标签必须是一维整数张量")

        unique_labels = set(labels.tolist())
        if not {0, 1}.issubset(unique_labels):
            raise ValueError(f"{task_name}的{split_name}标签缺少二分类类别 0 或 1")

    _validate_tensor(train_labels, "训练折")
    _validate_tensor(query_labels, "测试折")
