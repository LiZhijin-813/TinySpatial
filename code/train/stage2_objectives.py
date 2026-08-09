"""Stage 2 类别权重与任务损失。"""

from collections import Counter
from math import isfinite
from numbers import Real

import torch
import torch.nn as nn
import torch.nn.functional as functional


def compute_class_weights(
    samples,
    label_key,
    num_classes,
    ignore_index=None,
    device="cpu",
):
    """按类别频数计算加权交叉熵的静态权重。"""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("类别数必须为正整数")

    labels = []
    for sample in samples:
        label = int(sample[label_key])
        if ignore_index is not None and label == ignore_index:
            continue
        if not 0 <= label < num_classes:
            raise ValueError(
                f"标签 {label_key}={label} 超出 [0, {num_classes}) 范围"
            )
        labels.append(label)

    counts = Counter(labels)
    weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    total = len(labels)
    for class_index in range(num_classes):
        count = counts.get(class_index, 0)
        if count > 0:
            weights[class_index] = total / (num_classes * count)
    return weights


def compute_class_priors(
    samples,
    label_key,
    num_classes,
    ignore_index=None,
    device="cpu",
):
    """按训练样本频数计算严格为正的类别先验。"""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("类别数必须为正整数")

    counts = Counter()
    for sample in samples:
        label = int(sample[label_key])
        if ignore_index is not None and label == ignore_index:
            continue
        if not 0 <= label < num_classes:
            raise ValueError(
                f"标签 {label_key}={label} 超出 [0, {num_classes}) 范围"
            )
        counts[label] += 1

    total = sum(counts.values())
    if total == 0:
        raise ValueError("无法从空样本集合计算类别先验")
    missing = [index for index in range(num_classes) if counts[index] == 0]
    if missing:
        raise ValueError(f"类别先验存在空类别：{missing}")

    priors = torch.tensor(
        [counts[index] / total for index in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    return priors


class LogitAdjustedCrossEntropy(nn.Module):
    """在交叉熵前加入 tau 倍类别先验对数的分类损失。"""

    def __init__(self, class_priors, tau=1.0, reduction="mean"):
        super().__init__()
        priors = torch.as_tensor(class_priors, dtype=torch.float32)
        if priors.ndim != 1 or priors.numel() == 0:
            raise ValueError("class_priors 必须是一维非空张量")
        if not torch.isfinite(priors).all() or (priors <= 0).any():
            raise ValueError("class_priors 必须包含有限的正数")
        if not isinstance(tau, Real) or isinstance(tau, bool):
            raise ValueError("tau 必须为非布尔数值")
        if not isfinite(float(tau)) or tau < 0:
            raise ValueError("tau 必须为非负有限数")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction 必须为 none、mean 或 sum")

        self.register_buffer("log_priors", priors.log())
        self.tau = float(tau)
        self.reduction = reduction

    def forward(self, logits, targets):
        """返回加入类别先验校正后的交叉熵。"""
        adjusted_logits = logits + self.tau * self.log_priors
        return functional.cross_entropy(
            adjusted_logits,
            targets,
            reduction=self.reduction,
        )


def compute_stage2_losses(
    outputs,
    batch,
    task_mode,
    criteria,
    lambda_bm=0.3,
):
    """计算平坦分类或层级双头的加权交叉熵损失。"""
    if task_mode not in {"flat4", "flat5", "dual_head"}:
        raise ValueError(f"未知 task_mode: {task_mode}")
    if (
        not isinstance(lambda_bm, Real)
        or isinstance(lambda_bm, bool)
        or not isfinite(lambda_bm)
        or lambda_bm < 0
    ):
        raise ValueError("lambda_bm 必须为非负有限数")

    if task_mode == "flat4":
        loss = criteria["class"](outputs["class_logits"], batch["subtype_label"])
        return {"total_loss": loss, "class_loss": loss}

    if task_mode == "flat5":
        loss = criteria["class"](outputs["class_logits"], batch["class_label"])
        return {"total_loss": loss, "class_loss": loss}

    malignancy_loss = criteria["malignancy"](
        outputs["malignancy_logits"],
        batch["malignancy_label"],
    )
    malignant_mask = batch["malignancy_label"] == 1
    if malignant_mask.any():
        subtype_loss = criteria["subtype"](
            outputs["subtype_logits"][malignant_mask],
            batch["subtype_label"][malignant_mask],
        )
    else:
        subtype_loss = outputs["subtype_logits"].sum() * 0.0

    total_loss = subtype_loss + lambda_bm * malignancy_loss
    return {
        "total_loss": total_loss,
        "subtype_loss": subtype_loss,
        "malignancy_loss": malignancy_loss,
    }
