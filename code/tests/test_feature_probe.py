"""验证病例级融合特征诊断工具的纯函数语义。"""

import pytest
import torch

from code.train.feature_probe import (
    fit_linear_probe,
    nearest_centroid_predict,
)


def test_nearest_centroid_predict_returns_expected_labels():
    """最近类中心应将查询特征分配给欧氏距离最近的类别。"""
    train_features = torch.tensor([[0.0, 0.0], [0.2, 0.0], [3.0, 3.0], [3.2, 3.0]])
    train_labels = torch.tensor([0, 0, 1, 1])
    query_features = torch.tensor([[0.1, 0.1], [3.1, 2.8]])

    predictions = nearest_centroid_predict(
        train_features,
        train_labels,
        query_features,
        num_classes=2,
    )

    assert predictions.tolist() == [0, 1]


def test_nearest_centroid_predict_rejects_missing_training_class():
    """训练特征缺少目标类别时应显式报错。"""
    with pytest.raises(ValueError, match="类别"):
        nearest_centroid_predict(
            torch.zeros(2, 3),
            torch.tensor([0, 0]),
            torch.zeros(1, 3),
            num_classes=2,
        )


def test_fit_linear_probe_can_separate_linearly_separable_features():
    """线性可分的训练特征应能被轻量线性探针拟合。"""
    train_features = torch.tensor(
        [[-2.0, -1.0], [-1.5, -0.5], [1.5, 0.5], [2.0, 1.0]]
    )
    train_labels = torch.tensor([0, 0, 1, 1])
    query_features = torch.tensor([[-1.8, -0.7], [1.8, 0.7]])

    predictions = fit_linear_probe(
        train_features,
        train_labels,
        query_features,
        num_classes=2,
        epochs=200,
        learning_rate=0.1,
    )

    assert predictions.tolist() == [0, 1]


def test_fit_linear_probe_accepts_positive_class_weights():
    """线性探针应支持仅作用于训练损失的类别权重。"""
    train_features = torch.tensor(
        [[-2.0, -1.0], [-1.5, -0.5], [1.5, 0.5], [2.0, 1.0]]
    )
    train_labels = torch.tensor([0, 0, 1, 1])

    predictions = fit_linear_probe(
        train_features,
        train_labels,
        train_features,
        num_classes=2,
        epochs=20,
        learning_rate=0.1,
        class_weights=torch.tensor([1.0, 2.0]),
    )

    assert predictions.shape == train_labels.shape
