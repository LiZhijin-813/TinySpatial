"""提供病例级融合特征的轻量可分性诊断工具。"""

from math import isfinite

import torch
import torch.nn as nn


def _validate_feature_pair(features, labels, name):
    """校验特征矩阵与标签向量的基本形状和数值。"""
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(f"{name} 必须是二维特征张量")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"{name} 不能为空")
    if not torch.isfinite(features).all():
        raise ValueError(f"{name} 必须全部为有限数")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError(f"{name} 标签必须是一维张量")
    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"{name} 特征和标签数量不一致")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} 标签必须是整数张量")
    return features.float(), labels.long()


def _validate_query_features(query_features, feature_dim):
    """校验查询特征与训练特征的维度和数值。"""
    if not isinstance(query_features, torch.Tensor) or query_features.ndim != 2:
        raise ValueError("查询特征必须是二维张量")
    if query_features.shape[1] != feature_dim:
        raise ValueError("查询特征维度与训练特征不一致")
    if query_features.shape[0] == 0:
        raise ValueError("查询特征不能为空")
    if not torch.isfinite(query_features).all():
        raise ValueError("查询特征必须全部为有限数")
    return query_features.float()


def _validate_num_classes(num_classes):
    """校验类别数为正整数。"""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise ValueError("类别数必须是正整数")
    if num_classes <= 0:
        raise ValueError("类别数必须是正整数")


def _validate_labels_in_range(labels, num_classes, name):
    """校验标签位于指定类别空间内。"""
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError(f"{name} 标签超出类别范围")


def nearest_centroid_predict(
    train_features,
    train_labels,
    query_features,
    num_classes,
):
    """使用训练集类别中心对查询特征进行最近中心预测。"""
    _validate_num_classes(num_classes)
    train_features, train_labels = _validate_feature_pair(
        train_features,
        train_labels,
        "训练特征",
    )
    query_features = _validate_query_features(
        query_features,
        train_features.shape[1],
    )
    _validate_labels_in_range(train_labels, num_classes, "训练特征")

    centroids = []
    for class_index in range(num_classes):
        mask = train_labels == class_index
        if not mask.any():
            raise ValueError(f"训练特征缺少类别 {class_index}")
        centroids.append(train_features[mask].mean(dim=0))
    centroid_tensor = torch.stack(centroids)
    distances = torch.cdist(query_features, centroid_tensor)
    return distances.argmin(dim=1)


def fit_linear_probe(
    train_features,
    train_labels,
    query_features,
    num_classes,
    epochs=200,
    learning_rate=0.1,
    weight_decay=0.0,
    class_weights=None,
):
    """仅使用训练特征拟合线性探针，并返回查询特征的预测类别。"""
    _validate_num_classes(num_classes)
    train_features, train_labels = _validate_feature_pair(
        train_features,
        train_labels,
        "训练特征",
    )
    query_features = _validate_query_features(
        query_features,
        train_features.shape[1],
    )
    _validate_labels_in_range(train_labels, num_classes, "训练特征")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("训练轮数必须是正整数")
    if not isfinite(float(learning_rate)) or learning_rate <= 0:
        raise ValueError("学习率必须是正的有限数")
    if not isfinite(float(weight_decay)) or weight_decay < 0:
        raise ValueError("权重衰减必须是非负有限数")
    if class_weights is not None:
        if not isinstance(class_weights, torch.Tensor):
            raise ValueError("类别权重必须是一维张量")
        if class_weights.ndim != 1 or class_weights.shape[0] != num_classes:
            raise ValueError("类别权重维度必须与类别数一致")
        if not torch.isfinite(class_weights).all() or (class_weights <= 0).any():
            raise ValueError("类别权重必须是有限的正数")
        class_weights = class_weights.float()

    probe = nn.Linear(train_features.shape[1], num_classes)
    nn.init.zeros_(probe.weight)
    nn.init.zeros_(probe.bias)
    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(
            probe(train_features),
            train_labels,
            weight=class_weights,
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return probe(query_features).argmax(dim=1)
