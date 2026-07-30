"""通用分类评估工具和端到端预测语义。"""

import os
import re

import matplotlib
import numpy as np
import torch
from matplotlib import font_manager
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUBTYPE_NAMES = ["Luminal A", "Luminal B", "HER2+", "TNBC"]
_CHINESE_FONT_NAMES = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "Droid Sans Fallback",
]


def _validate_class_names(class_names):
    """校验类别名称非空且不重复。"""
    try:
        names = SUBTYPE_NAMES if class_names is None else list(class_names)
    except TypeError as error:
        raise ValueError("class_names 必须是非空类别名称集合") from error
    if not names or len(set(names)) != len(names):
        raise ValueError("class_names 必须非空且唯一")
    return names


def _validate_label_array(values, name, class_count):
    """校验标签数组形状、类型、长度和取值范围。"""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} 必须是一维非空数组")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} 必须是整数标签")
    if np.any(array < 0) or np.any(array >= class_count):
        raise ValueError(f"{name} 必须位于 0 到类别数减一之间")
    return array


def _validate_focus_labels(focus_labels, class_count):
    """校验关注类别集合是合法的非空标签子集。"""
    try:
        labels = list(focus_labels)
    except TypeError as error:
        raise ValueError("focus_labels 必须是非空标签集合") from error
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("focus_labels 必须非空且唯一")
    if any(
        isinstance(label, bool)
        or not isinstance(label, (int, np.integer))
        or label < 0
        or label >= class_count
        for label in labels
    ):
        raise ValueError("focus_labels 必须是合法标签子集")
    return labels


def _validate_scores(y_true, y_score, class_count):
    """校验二分类 AUC 所需的一维正类分数。"""
    if class_count != 2:
        raise ValueError("y_score 仅允许用于二分类评估")
    scores = np.asarray(y_score)
    if scores.ndim != 1 or len(scores) != len(y_true):
        raise ValueError("y_score 必须是一维且与 y_true 等长")
    try:
        finite = np.isfinite(scores)
    except TypeError as error:
        raise ValueError("y_score 必须是有限数值") from error
    if not np.all(finite):
        raise ValueError("y_score 必须是有限数值")
    if not np.any(y_true == 0) or not np.any(y_true == 1):
        raise ValueError("y_true 必须同时包含二分类标签 0 和 1")
    return scores


def evaluate_predictions(
    y_true,
    y_pred,
    class_names=None,
    focus_labels=None,
    y_score=None,
    save_dir=None,
    figure_name="confusion_matrix.png",
):
    """按配置类别计算分类指标，可将宏平均限制在关注类别。"""
    class_names = _validate_class_names(class_names)
    labels = list(range(len(class_names)))
    y_true = _validate_label_array(y_true, "y_true", len(labels))
    y_pred = _validate_label_array(y_pred, "y_pred", len(labels))
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 必须等长")
    if focus_labels is None:
        focus_labels = labels
    else:
        focus_labels = _validate_focus_labels(focus_labels, len(labels))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for index, name in enumerate(class_names):
        true_positive = int(((y_true == index) & (y_pred == index)).sum())
        false_negative = int(((y_true == index) & (y_pred != index)).sum())
        false_positive = int(((y_true != index) & (y_pred == index)).sum())
        true_negative = int(((y_true != index) & (y_pred != index)).sum())
        per_class[name] = {
            "recall": true_positive / max(true_positive + false_negative, 1),
            "precision": true_positive / max(true_positive + false_positive, 1),
            "specificity": true_negative / max(true_negative + false_positive, 1),
            "f1": (
                2 * true_positive
                / max(2 * true_positive + false_positive + false_negative, 1)
            ),
        }

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            recall_score(
                y_true,
                y_pred,
                labels=focus_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, labels=focus_labels, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=focus_labels, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, labels=focus_labels, average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "label_distribution": np.bincount(
            y_true, minlength=len(class_names)
        ).tolist(),
        "prediction_distribution": np.bincount(
            y_pred, minlength=len(class_names)
        ).tolist(),
    }
    if y_score is not None:
        scores = _validate_scores(y_true, y_score, len(class_names))
        result["auc"] = float(roc_auc_score(y_true, scores))
    if save_dir:
        _plot_confusion_matrix(cm, class_names, save_dir, figure_name)
    return result


def conditional_malignant_predictions(class_logits):
    """忽略良性 logit，仅在四个恶性亚型中决策。"""
    return class_logits[:, :4].argmax(dim=1).detach().cpu().numpy()


def end_to_end_flat5_predictions(class_logits):
    """执行五分类端到端预测，类别 4 表示预测为良性。"""
    return class_logits.argmax(dim=1).detach().cpu().numpy()


def end_to_end_dual_predictions(malignancy_logits, subtype_logits):
    """执行双头端到端预测，二分类良性结果映射为类别 4。"""
    binary = malignancy_logits.argmax(dim=1)
    subtype = subtype_logits.argmax(dim=1)
    combined = torch.where(binary == 0, torch.full_like(subtype, 4), subtype)
    return combined.detach().cpu().numpy()


def _find_chinese_font():
    """按优先级查找可用的中文字体。"""
    for font_path in sorted(font_manager.findSystemFonts()):
        normalized_name = re.sub(
            r"[^a-z0-9]", "", os.path.basename(font_path).lower()
        )
        if "notosanscjk" in normalized_name:
            return font_manager.FontProperties(fname=font_path)
    for font_name in _CHINESE_FONT_NAMES:
        try:
            font_path = font_manager.findfont(
                font_manager.FontProperties(family=font_name),
                fallback_to_default=False,
            )
        except (OSError, ValueError):
            continue
        return font_manager.FontProperties(fname=font_path)
    return None


def _plot_confusion_matrix(cm, class_names, save_dir, figure_name):
    """使用配置类别名称和可用中文字体绘制混淆矩阵。"""
    os.makedirs(save_dir, exist_ok=True)
    chinese_font = _find_chinese_font()
    latin_font = font_manager.FontProperties(family="DejaVu Sans")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontproperties=latin_font)
    ax.set_yticklabels(class_names, fontproperties=latin_font)
    ax.set_title("混淆矩阵", fontproperties=chinese_font)
    ax.set_xlabel("预测类别", fontproperties=chinese_font)
    ax.set_ylabel("真实类别", fontproperties=chinese_font)

    threshold = cm.max() / 2.0
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            ax.text(
                column,
                row,
                format(cm[row, column], "d"),
                ha="center",
                va="center",
                color="white" if cm[row, column] > threshold else "black",
                fontproperties=latin_font,
            )
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, figure_name), dpi=150)
    plt.close(fig)


def print_evaluation(results):
    """按结果中的动态类别打印评估报告。"""
    print("\n" + "=" * 60)
    print("  分类评估报告")
    print("=" * 60)
    print(f"  总体准确率: {results['accuracy']:.4f}")
    print(f"  宏平均 F1: {results['macro_f1']:.4f}")
    print(f"  宏平均召回率: {results['macro_recall']:.4f}")
    print(f"  宏平均精确率: {results['macro_precision']:.4f}")
    print("\n  类别              召回率    精确率    特异度        F1")
    print("  " + "-" * 52)
    for name, metrics in results["per_class"].items():
        print(
            f"  {name:<16} {metrics['recall']:>7.4f}"
            f" {metrics['precision']:>9.4f}"
            f" {metrics['specificity']:>9.4f}"
            f" {metrics['f1']:>9.4f}"
        )
    print("=" * 60 + "\n")
