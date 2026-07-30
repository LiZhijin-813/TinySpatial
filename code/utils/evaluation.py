"""通用分类评估工具和端到端预测语义。"""

import os

import matplotlib
import numpy as np
import torch
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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_names = SUBTYPE_NAMES if class_names is None else list(class_names)
    labels = list(range(len(class_names)))
    focus_labels = labels if focus_labels is None else list(focus_labels)
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
                labels=focus_labels,
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
        result["auc"] = float(roc_auc_score(y_true, np.asarray(y_score)))
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


def _plot_confusion_matrix(cm, class_names, save_dir, figure_name):
    """使用配置类别名称绘制并保存混淆矩阵。"""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="混淆矩阵",
        xlabel="预测类别",
        ylabel="真实类别",
    )

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
