import pytest
import torch

from code.train.swe_full_test_shuffle import (
    build_case_comparison_rows,
    build_full_shuffle_indices,
    predict_malignant_logits,
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


def test_malignant_prediction_ignores_high_fifth_class_logit():
    logits = torch.tensor([[0.0, 1.0, 0.0, 0.0, 100.0]])

    predicted, confidence = predict_malignant_logits(logits)

    expected_confidence = torch.softmax(logits[:, :4], dim=1)[0, 1].item()
    assert predicted.tolist() == [1]
    assert confidence.tolist() == pytest.approx([expected_confidence])


def test_case_comparison_rejects_duplicate_case_ids():
    base_rows = [
        {"case_id": "a", "true_label": 0, "predicted_label": 0, "confidence": 0.8},
        {"case_id": "b", "true_label": 1, "predicted_label": 1, "confidence": 0.8},
    ]
    shuffled_rows = [
        {"case_id": "a", "true_label": 0, "predicted_label": 0, "confidence": 0.7},
        {"case_id": "a", "true_label": 0, "predicted_label": 1, "confidence": 0.6},
    ]

    with pytest.raises(ValueError, match="病例编号必须唯一"):
        build_case_comparison_rows(base_rows, shuffled_rows)


def test_case_comparison_aligns_shuffled_order_to_base_order():
    base_rows = [
        {"case_id": "a", "true_label": 0, "predicted_label": 0, "confidence": 0.8},
        {"case_id": "b", "true_label": 1, "predicted_label": 1, "confidence": 0.8},
    ]
    shuffled_rows = [
        {"case_id": "b", "true_label": 1, "predicted_label": 0, "confidence": 0.7},
        {"case_id": "a", "true_label": 0, "predicted_label": 1, "confidence": 0.6},
    ]
    shuffled_rows[0].update({"donor_case_id": "a", "donor_label": 0})
    shuffled_rows[1].update({"donor_case_id": "b", "donor_label": 1})

    result = build_case_comparison_rows(base_rows, shuffled_rows)

    assert [row["case_id"] for row in result] == ["a", "b"]
    assert [row["shuffled_predicted_label"] for row in result] == [1, 0]
    assert [row["donor_case_id"] for row in result] == ["b", "a"]


def test_case_comparison_rejects_case_id_set_mismatch():
    base_rows = [
        {"case_id": "a", "true_label": 0, "predicted_label": 0, "confidence": 0.8},
        {"case_id": "b", "true_label": 1, "predicted_label": 1, "confidence": 0.8},
    ]
    shuffled_rows = [
        {"case_id": "a", "true_label": 0, "predicted_label": 0, "confidence": 0.7},
        {"case_id": "c", "true_label": 1, "predicted_label": 1, "confidence": 0.6},
    ]

    with pytest.raises(ValueError, match="病例编号集合不一致"):
        build_case_comparison_rows(base_rows, shuffled_rows)


def test_case_comparison_rejects_true_label_mismatch():
    base_rows = [{"case_id": "a", "true_label": 0, "predicted_label": 0, "confidence": 0.8}]
    shuffled_rows = [{"case_id": "a", "true_label": 1, "predicted_label": 0, "confidence": 0.7}]

    with pytest.raises(ValueError, match="真实标签不一致"):
        build_case_comparison_rows(base_rows, shuffled_rows)


def test_case_comparison_rejects_self_pairing_and_invalid_donor_label():
    base_rows = [{"case_id": "a", "true_label": 0, "predicted_label": 0, "confidence": 0.8}]
    self_paired = [
        {
            "case_id": "a",
            "true_label": 0,
            "predicted_label": 0,
            "confidence": 0.7,
            "donor_case_id": "a",
            "donor_label": 0,
        }
    ]
    invalid_label = [
        {
            "case_id": "a",
            "true_label": 0,
            "predicted_label": 0,
            "confidence": 0.7,
            "donor_case_id": "b",
            "donor_label": "0",
        }
    ]

    with pytest.raises(ValueError, match="供体病例不能与当前病例相同"):
        build_case_comparison_rows(base_rows, self_paired)
    with pytest.raises(ValueError, match="供体标签必须是整数"):
        build_case_comparison_rows(base_rows, invalid_label)


def test_metrics_keep_fixed_four_class_shape():
    rows = [
        {"true_label": 0, "base_predicted_label": 0, "shuffled_predicted_label": 1},
        {"true_label": 1, "base_predicted_label": 1, "shuffled_predicted_label": 1},
    ]

    summary = summarize_condition_metrics(rows)

    assert len(summary["base"]["confusion_matrix"]) == 4
    assert len(summary["shuffled"]["confusion_matrix"]) == 4
    assert len(summary["base"]["per_class_f1"]) == 4


def test_metrics_include_reproducible_values_and_all_required_keys():
    rows = [
        {"true_label": 0, "base_predicted_label": 0, "shuffled_predicted_label": 1},
        {"true_label": 1, "base_predicted_label": 1, "shuffled_predicted_label": 1},
    ]

    summary = summarize_condition_metrics(rows)

    assert set(summary) == {"base", "shuffled"}
    assert summary["base"]["accuracy"] == pytest.approx(1.0)
    assert summary["shuffled"]["accuracy"] == pytest.approx(0.5)
    assert summary["base"]["per_class_f1"] == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert summary["base"]["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert summary["base"]["prediction_distribution"] == [1, 1, 0, 0]
