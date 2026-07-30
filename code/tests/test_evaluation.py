import numpy as np
import pytest
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


@pytest.mark.parametrize(
    "y_true, y_pred, class_names",
    [
        (np.array([[0, 1]]), np.array([0, 1]), ["A", "B"]),
        (np.array([0, 1]), np.array([[0, 1]]), ["A", "B"]),
        (np.array([0]), np.array([0, 1]), ["A", "B"]),
        (np.array([]), np.array([]), ["A", "B"]),
        (np.array([-1, 0]), np.array([0, 0]), ["A", "B"]),
        (np.array([0, 2]), np.array([0, 0]), ["A", "B"]),
        (np.array([0, 1]), np.array([-1, 0]), ["A", "B"]),
        (np.array([0, 1]), np.array([0, 2]), ["A", "B"]),
    ],
)
def test_invalid_labels_and_shapes_raise_value_error(y_true, y_pred, class_names):
    """输入必须是一维、等长且落在配置类别范围内。"""
    with pytest.raises(ValueError):
        evaluate_predictions(y_true, y_pred, class_names=class_names)


def test_prediction_distributions_keep_configured_class_count():
    """标签分布和预测分布的长度必须等于配置类别数。"""
    result = evaluate_predictions(
        np.array([0, 1]),
        np.array([1, 1]),
        class_names=["A", "B", "C"],
    )
    assert len(result["label_distribution"]) == 3
    assert len(result["prediction_distribution"]) == 3


def test_focus_labels_limit_macro_metrics_but_not_weighted_f1():
    """focus_labels 只影响宏指标，weighted F1 必须覆盖全部配置类别。"""
    result = evaluate_predictions(
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([0, 1, 1, 1, 1, 0]),
        class_names=["A", "B"],
        focus_labels=[1],
    )
    assert result["balanced_accuracy"] == pytest.approx(2 / 3)
    assert result["macro_precision"] == pytest.approx(1 / 2)
    assert result["macro_recall"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(4 / 7)
    assert result["weighted_f1"] == pytest.approx(17 / 35)


@pytest.mark.parametrize(
    "y_true, y_score, class_names",
    [
        (np.array([0, 1, 2, 3]), np.array([0.1, 0.2, 0.3, 0.4]), SUBTYPES),
        (
            np.array([0, 0, 1, 1]),
            np.array([[0.1], [0.2], [0.8], [0.9]]),
            ["Benign", "Malignant"],
        ),
        (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8]), ["Benign", "Malignant"]),
        (np.array([0, 0]), np.array([0.1, 0.2]), ["Benign", "Malignant"]),
        (
            np.array([0, 0, 1, 1]),
            np.array([0.1, np.nan, 0.8, 0.9]),
            ["Benign", "Malignant"],
        ),
    ],
)
def test_invalid_auc_scores_raise_value_error(y_true, y_score, class_names):
    """AUC 分数必须是二分类、等长、一维且有限的正类分数。"""
    y_pred = np.zeros_like(y_true)
    with pytest.raises(ValueError):
        evaluate_predictions(
            y_true,
            y_pred,
            class_names=class_names,
            y_score=y_score,
        )


def test_save_dir_keyword_uses_default_classes_and_requested_figure_name(tmp_path):
    """旧式 save_dir 关键字调用应使用默认四类名并生成指定图片。"""
    figure_name = "evaluation-custom.png"
    result = evaluate_predictions(
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 1, 3]),
        save_dir=tmp_path,
        figure_name=figure_name,
    )
    assert set(result["per_class"]) == set(SUBTYPES)
    assert (tmp_path / figure_name).is_file()
