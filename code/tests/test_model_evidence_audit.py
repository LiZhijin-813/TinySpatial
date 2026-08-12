import numpy as np
import pytest

from code.train.model_evidence_audit import (
    compare_prediction_records,
    compute_spatial_concentration,
    mask_text_terms,
    summarize_saliency,
    summarize_evidence,
)


def test_mask_text_terms_replaces_only_requested_terms():
    result = mask_text_terms("Luminal B，HER2 阴性", ["Luminal B"])

    assert result == "[遮蔽]，HER2 阴性"


def test_compute_spatial_concentration_returns_center_and_edge_ratio():
    saliency = np.zeros((4, 4), dtype=np.float32)
    saliency[1:3, 1:3] = 1.0

    result = compute_spatial_concentration(saliency, center_fraction=0.5)

    assert result["center_ratio"] == pytest.approx(1.0)
    assert result["edge_ratio"] == pytest.approx(0.0)


def test_compare_prediction_records_marks_changed_case():
    base = [{"case_id": "a", "predicted_label": 0, "confidence": 0.8}]
    ablated = [{"case_id": "a", "predicted_label": 2, "confidence": 0.6}]

    result = compare_prediction_records(base, ablated)

    assert result[0]["prediction_changed"] is True
    assert result[0]["confidence_delta"] == pytest.approx(-0.2)


def test_compare_prediction_records_rejects_mismatched_case_ids():
    with pytest.raises(ValueError, match="病例编号"):
        compare_prediction_records(
            [{"case_id": "a", "predicted_label": 0, "confidence": 0.8}],
            [{"case_id": "b", "predicted_label": 0, "confidence": 0.8}],
        )


def test_summarize_evidence_counts_changed_predictions():
    rows = [
        {
            "case_id": "a",
            "true_label": 1,
            "error_type": "亚型错分",
            "modality": "swe",
            "prediction_changed": True,
            "confidence_delta": -0.2,
        },
        {
            "case_id": "b",
            "true_label": 3,
            "error_type": "正确",
            "modality": "swe",
            "prediction_changed": False,
            "confidence_delta": 0.0,
        },
    ]

    summary = summarize_evidence(rows)

    assert summary["row_count"] == 2
    assert summary["changed_prediction_count"] == 1
    assert summary["by_modality"]["swe"]["changed_prediction_count"] == 1


def test_summarize_saliency_aggregates_modalities_without_raw_maps():
    summary = summarize_saliency(
        [
            {"case_id": "a", "modality": "bus", "center_ratio": 0.4, "edge_ratio": 0.6},
            {"case_id": "b", "modality": "bus", "center_ratio": 0.6, "edge_ratio": 0.4},
        ]
    )

    assert summary["bus"]["row_count"] == 2
    assert summary["bus"]["mean_center_ratio"] == pytest.approx(0.5)
