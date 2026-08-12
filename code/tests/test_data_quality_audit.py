import json

import numpy as np
from PIL import Image

from code.train.data_quality_audit import (
    audit_image_file,
    build_quality_summary,
    classify_text_risk,
)


def test_classify_text_risk_marks_target_terms():
    result = classify_text_risk("报告提示 Luminal B，未见 TNBC")

    assert result["target_keyword_hits"] == ["Luminal B", "TNBC"]
    assert result["contains_target_keyword"] is True


def test_classify_text_risk_does_not_return_raw_text():
    result = classify_text_risk("患者文本中的隐私内容")

    assert "raw_text" not in result
    assert result["text_length"] == 10


def test_audit_image_file_marks_missing_file(tmp_path):
    result = audit_image_file(tmp_path / "missing.png")

    assert result["exists"] is False
    assert result["readable"] is False


def test_audit_image_file_marks_near_constant_image(tmp_path):
    image_path = tmp_path / "constant.png"
    Image.fromarray(np.full((8, 10, 3), 128, dtype=np.uint8)).save(image_path)

    result = audit_image_file(image_path)

    assert result["exists"] is True
    assert result["readable"] is True
    assert result["width"] == 10
    assert result["height"] == 8
    assert result["near_constant"] is True


def test_build_quality_summary_counts_labels_and_risks():
    rows = [
        {
            "case_id": "a-1",
            "subtype_label": 0,
            "group_id": "a",
            "bus_exists": True,
            "swe_exists": True,
            "cdfi_exists": True,
            "text_exists": True,
            "contains_target_keyword": False,
            "label_conflict": False,
        },
        {
            "case_id": "a-2",
            "subtype_label": 1,
            "group_id": "a",
            "bus_exists": True,
            "swe_exists": False,
            "cdfi_exists": True,
            "text_exists": True,
            "contains_target_keyword": True,
            "label_conflict": True,
        },
    ]

    summary = build_quality_summary(rows, rows)

    assert summary["label_counts"] == [1, 1, 0, 0]
    assert summary["multi_case_group_count"] == 1
    assert summary["target_keyword_case_count"] == 1
    assert summary["label_conflict_case_count"] == 1


def test_quality_summary_is_json_serializable():
    row = {
        "case_id": "a-1",
        "subtype_label": 0,
        "group_id": "a",
        "bus_exists": True,
        "swe_exists": True,
        "cdfi_exists": True,
        "text_exists": True,
        "contains_target_keyword": False,
        "label_conflict": False,
    }

    json.dumps(build_quality_summary([row], [row]), ensure_ascii=False)
