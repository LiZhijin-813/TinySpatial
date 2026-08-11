"""层级探针的任务配置与标签筛选纯函数。"""

from __future__ import annotations

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
