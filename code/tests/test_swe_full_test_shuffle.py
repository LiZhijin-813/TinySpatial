import pytest

from code.train.swe_full_test_shuffle import (
    build_full_shuffle_indices,
    summarize_condition_metrics,
)


def test_full_shuffle_is_deterministic_and_has_no_self_pairing():
    case_ids = ["a", "b", "c"]
    indices = build_full_shuffle_indices(case_ids)

    assert indices == [1, 2, 0]
    assert all(index != position for position, index in enumerate(indices))


def test_full_shuffle_rejects_single_case():
    with pytest.raises(ValueError, match="至少需要两个病例"):
        build_full_shuffle_indices(["a"])


def test_metrics_keep_fixed_four_class_shape():
    rows = [
        {"true_label": 0, "base_predicted_label": 0, "shuffled_predicted_label": 1},
        {"true_label": 1, "base_predicted_label": 1, "shuffled_predicted_label": 1},
    ]

    summary = summarize_condition_metrics(rows)

    assert len(summary["base"]["confusion_matrix"]) == 4
    assert len(summary["shuffled"]["confusion_matrix"]) == 4
    assert len(summary["base"]["per_class_f1"]) == 4
