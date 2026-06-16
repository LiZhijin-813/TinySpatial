"""临床评估管线：面向乳腺癌分子亚型的多维度性能评估。

提供全局 Accuracy、逐类 Recall/Sensitivity、Specificity、Precision、
F1-Score 以及混淆矩阵可视化。特别关注 HER2+ 与 TNBC 的分型指标。

评估指标体系：
    - 全局准确率 (Accuracy)
    - 逐类灵敏度 (Recall / Sensitivity)
    - 逐类特异度 (Specificity)
    - 逐类阳性预测值 (Precision)
    - 逐类 F1-Score
    - 混淆矩阵 (Confusion Matrix)：分析各亚型间误判流向
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")  # 无头后端，适用于服务器环境
import matplotlib.pyplot as plt
import os

SUBTYPE_NAMES = ["Luminal A", "Luminal B", "HER2+", "TNBC"]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str = None) -> dict:
    """计算全面的临床评估指标。

    Args:
        y_true: (N,) 真实标签 (0-3)
        y_pred: (N,) 预测标签 (0-3)
        save_dir: 混淆矩阵图保存目录，为 None 则不保存

    Returns:
        包含全局指标、逐类指标、混淆矩阵的字典
    """
    # 全局准确率
    acc = accuracy_score(y_true, y_pred)

    # 逐类指标
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)      # 灵敏度
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # 逐类特异度: TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    specificity = []
    for i in range(4):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp + 1e-8))

    # 宏平均
    macro_recall = recall.mean()
    macro_precision = precision.mean()
    macro_f1 = f1.mean()
    macro_specificity = np.mean(specificity)

    # 加权平均
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    results = {
        "accuracy": acc,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "macro_specificity": macro_specificity,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": {},
        "confusion_matrix": cm.tolist(),
    }

    # 逐类汇总
    for i, name in enumerate(SUBTYPE_NAMES):
        results["per_class"][name] = {
            "recall": float(recall[i]),
            "precision": float(precision[i]),
            "specificity": float(specificity[i]),
            "f1": float(f1[i]),
        }

    # HER2+ 与 TNBC 重点关注
    results["HER2_plus"] = results["per_class"]["HER2+"]
    results["TNBC"] = results["per_class"]["TNBC"]

    # 保存混淆矩阵图
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _plot_confusion_matrix(cm, save_dir)

    return results


def _plot_confusion_matrix(cm: np.ndarray, save_dir: str):
    """绘制并保存混淆矩阵图。

    Args:
        cm: (4, 4) 混淆矩阵
        save_dir: 保存目录
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=SUBTYPE_NAMES,
        yticklabels=SUBTYPE_NAMES,
        title="Confusion Matrix",
        xlabel="Predicted",
        ylabel="True",
    )

    # 在每个单元格中标注数值
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def print_evaluation(results: dict):
    """格式化打印评估结果。

    Args:
        results: evaluate_predictions 返回的结果字典
    """
    print(f"\n{'='*60}")
    print(f"  临床评估报告 (Clinical Evaluation Report)")
    print(f"{'='*60}")
    print(f"  总体准确率 (Accuracy):  {results['accuracy']:.4f}")
    print(f"  宏平均 F1 (Macro F1):   {results['macro_f1']:.4f}")
    print(f"  宏平均灵敏度 (Recall):  {results['macro_recall']:.4f}")
    print(f"  宏平均精确率 (Prec):    {results['macro_precision']:.4f}")
    print(f"  宏平均特异度 (Spec):    {results['macro_specificity']:.4f}")
    print(f"\n  {'类别':<12} {'灵敏度':>8} {'精确率':>8} {'特异度':>8} {'F1':>8}")
    print(f"  {'-'*48}")
    for name in SUBTYPE_NAMES:
        c = results["per_class"][name]
        print(f"  {name:<12} {c['recall']:>8.4f} {c['precision']:>8.4f} {c['specificity']:>8.4f} {c['f1']:>8.4f}")
    print(f"\n  [重点关注] HER2+ 灵敏度: {results['HER2_plus']['recall']:.4f} | TNBC 灵敏度: {results['TNBC']['recall']:.4f}")
    print(f"{'='*60}\n")
