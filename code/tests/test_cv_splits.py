"""验证病例组约束分层交叉验证划分工具。"""

import pytest

from code.train.cv_splits import build_group_stratified_folds


def _samples():
    return [
        {"case_id": "a-1", "subtype_label": 0},
        {"case_id": "a-2", "subtype_label": 0},
        {"case_id": "b-1", "subtype_label": 1},
        {"case_id": "c-1", "subtype_label": 1},
        {"case_id": "d-1", "subtype_label": 2},
        {"case_id": "e-1", "subtype_label": 2},
        {"case_id": "f-1", "subtype_label": 3},
        {"case_id": "g-1", "subtype_label": 3},
    ]


def test_group_stratified_folds_keep_same_prefix_together():
    """同一前缀的病例必须进入同一折。"""
    folds = build_group_stratified_folds(_samples(), n_splits=2, seed=42)

    locations = {}
    for fold_index, fold in enumerate(folds):
        for sample in fold:
            group_id = sample["case_id"].split("-", maxsplit=1)[0]
            locations.setdefault(group_id, set()).add(fold_index)

    assert all(len(indices) == 1 for indices in locations.values())


def test_group_stratified_folds_cover_each_sample_once():
    """所有输入病例应恰好出现在一个验证折中。"""
    folds = build_group_stratified_folds(_samples(), n_splits=2, seed=42)
    fold_ids = [sample["case_id"] for fold in folds for sample in fold]

    assert sorted(fold_ids) == sorted(sample["case_id"] for sample in _samples())
    assert all(folds)
    assert max(len(fold) for fold in folds) - min(len(fold) for fold in folds) <= 2


def test_group_stratified_folds_reject_too_many_folds_for_a_class():
    """折数超过最少类别样本数时应显式报错。"""
    with pytest.raises(ValueError, match="类别"):
        build_group_stratified_folds(_samples(), n_splits=3, seed=42)
