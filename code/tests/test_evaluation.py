import numpy as np
import torch

from code.utils.evaluation import (
    conditional_malignant_predictions,
    end_to_end_dual_predictions,
    end_to_end_flat5_predictions,
    evaluate_predictions,
)


SUBTYPES = ["Luminal A", "Luminal B", "HER2+", "TNBC"]


def test_four_class_metrics_have_configured_dimensions():
    """四分类指标应按传入的类别配置计算并保留四个维度。"""
    result = evaluate_predictions(
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 1, 3]),
        class_names=SUBTYPES,
    )
    assert result["accuracy"] == 0.75
    assert len(result["confusion_matrix"]) == 4
    assert set(result["per_class"]) == set(SUBTYPES)
    assert "balanced_accuracy" in result
    assert result["prediction_distribution"] == [1, 2, 0, 1]


def test_flat5_conditional_and_end_to_end_predictions_differ():
    """flat5 条件预测与端到端预测应表达不同的决策语义。"""
    logits = torch.tensor([
        [4.0, 1.0, 0.0, 0.0, 5.0],
        [0.0, 4.0, 1.0, 0.0, 2.0],
    ])
    assert conditional_malignant_predictions(logits).tolist() == [0, 1]
    assert end_to_end_flat5_predictions(logits).tolist() == [4, 1]


def test_dual_head_end_to_end_marks_binary_benign_as_class_four():
    """dual_head 端到端预测中，二分类良性应映射为类别 4。"""
    malignancy_logits = torch.tensor([[5.0, 1.0], [1.0, 5.0]])
    subtype_logits = torch.tensor([
        [0.0, 0.0, 5.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
    ])
    assert end_to_end_dual_predictions(
        malignancy_logits, subtype_logits
    ).tolist() == [4, 1]


def test_binary_auc_is_reported():
    """二分类评估应报告 AUC，并计算阳性类召回率。"""
    result = evaluate_predictions(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        class_names=["Benign", "Malignant"],
        y_score=np.array([0.1, 0.6, 0.8, 0.9]),
    )
    assert result["auc"] == 1.0
    assert result["per_class"]["Malignant"]["recall"] == 1.0
