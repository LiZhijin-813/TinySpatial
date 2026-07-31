"""Stage 2 类别权重与任务损失。"""

from collections import Counter
from math import isfinite
from numbers import Real

import torch


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
