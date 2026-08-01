"""验证病例级错误审计的可序列化领域产物。"""

import csv
import json
import math

import pytest
import torch

from code.train.case_audit import (
    build_audit_summary,
    build_case_records,
    validate_output_directory,
    write_audit_outputs,
)
from code.utils.evaluation import SUBTYPE_NAMES, evaluate_predictions


def _records():
    """构造覆盖正确、亚型错分与端到端良性预测的最小病例集合。"""
    return build_case_records(
        ["病例-甲", "病例-乙", "病例-丙"],
        [0, 1, 2],
        torch.tensor(
            [
                [4.0, 1.0, 0.0, -1.0, -2.0],
                [3.0, 0.0, 2.0, -1.0, -2.0],
                [0.0, 1.0, 2.0, -1.0, 5.0],
            ]
        ),
        [
            [True, True, True, True],
            [True, False, True, True],
            [False, True, True, False],
        ],
    )


def _saved_metrics(records):
    """使用既有评估器构造应当被审计模块复现的保存指标。"""
    return {
        "malignant": evaluate_predictions(
            [record["true_subtype_label"] for record in records],
            [record["conditional_predicted_subtype_label"] for record in records],
            SUBTYPE_NAMES,
        )
    }


def test_build_case_records_preserves_order_and_audits_error_types():
    """病例顺序和三种审计结论必须符合 flat5 语义。"""
    records = _records()

    assert [record["case_id"] for record in records] == ["病例-甲", "病例-乙", "病例-丙"]
    assert [record["error_type"] for record in records] == [
        "正确",
        "亚型错分",
        "恶性病例预测为良性",
    ]
    assert records[0]["true_subtype_name"] == "Luminal A"
    assert records[1]["conditional_predicted_subtype_name"] == "Luminal A"
    assert records[2]["end_to_end_predicted_class_name"] == "良性"


def test_case_record_probabilities_are_finite_and_independently_normalized():
    """五分类和条件四分类概率必须分别是有限且归一的浮点数。"""
    for record in _records():
        for field, expected_size in (("five_class_probabilities", 5), ("conditional_probabilities", 4)):
            probabilities = record[field]
            assert len(probabilities) == expected_size
            assert all(isinstance(value, float) and math.isfinite(value) for value in probabilities)
            assert sum(probabilities) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "case_ids, labels, logits, modalities",
    [
        ([], [], torch.empty((0, 5)), []),
        (["重复", "重复"], [0, 1], torch.zeros((2, 5)), [[True] * 4] * 2),
        (["甲"], [0], torch.zeros((1, 4)), [[True] * 4]),
        (["甲"], [0, 1], torch.zeros((1, 5)), [[True] * 4]),
        (["甲"], [0], torch.tensor([[0.0, 0.0, 0.0, 0.0, float("nan")]]), [[True] * 4]),
        (["甲"], [0], torch.zeros((1, 5)), [[True] * 3]),
    ],
)
def test_build_case_records_rejects_invalid_inputs(case_ids, labels, logits, modalities):
    """记录构造必须拒绝不完整、非有限或不一致的输入。"""
    with pytest.raises(ValueError, match="[\u4e00-\u9fff]"):
        build_case_records(case_ids, labels, logits, modalities)


def test_build_audit_summary_reproduces_saved_malignant_metrics():
    """摘要必须复用评估器并复现保存的恶性区域指标。"""
    records = _records()
    expected = _saved_metrics(records)
    summary = build_audit_summary(records, expected)

    assert summary["malignant"] == expected["malignant"]
    assert summary["error_type_distribution"] == {"正确": 1, "亚型错分": 1, "恶性病例预测为良性": 1}


def test_build_audit_summary_rejects_unreproducible_saved_metrics():
    """保存指标与病例记录不一致时必须明确拒绝。"""
    with pytest.raises(ValueError, match="无法复现"):
        build_audit_summary(_records(), {"malignant": {"accuracy": 0.0}})


def test_validate_output_directory_rejects_nonempty_directory_unless_overwritten(tmp_path):
    """非空输出目录只有显式覆盖时才允许继续写入。"""
    output_dir = tmp_path / "审计"
    assert validate_output_directory(output_dir, overwrite=False) == output_dir
    (output_dir / "已有.txt").write_text("内容", encoding="utf-8")

    with pytest.raises(ValueError, match="[\u4e00-\u9fff]"):
        validate_output_directory(output_dir, overwrite=False)
    assert validate_output_directory(output_dir, overwrite=True) == output_dir


def test_write_audit_outputs_rejects_nonempty_directory_unless_overwritten(tmp_path):
    """公开写入接口必须拒绝非空目录，只有显式覆盖时才允许写入。"""
    output_dir = tmp_path / "审计"
    output_dir.mkdir()
    (output_dir / "已有.txt").write_text("内容", encoding="utf-8")
    records = _records()
    summary = build_audit_summary(records, _saved_metrics(records))

    with pytest.raises(ValueError, match="[\u4e00-\u9fff]"):
        write_audit_outputs(output_dir, records, summary)

    write_audit_outputs(output_dir, records, summary, overwrite=True)

    assert (output_dir / "case_predictions.csv").exists()


def test_build_audit_summary_uses_chinese_error_for_missing_saved_metrics():
    """公开异常不得暴露英文保存指标或恶性区域内部字段名。"""
    with pytest.raises(ValueError) as error:
        build_audit_summary(_records(), {})

    assert "保存指标" in str(error.value)
    assert "saved_metrics" not in str(error.value)
    assert "malignant" not in str(error.value)


def test_write_audit_outputs_writes_three_files_and_no_high_confidence_branch(tmp_path):
    """审计产物必须按记录顺序写 CSV，并说明无高置信度错分。"""
    records = _records()
    summary = build_audit_summary(records, _saved_metrics(records))

    write_audit_outputs(tmp_path, records, summary)

    csv_path = tmp_path / "case_predictions.csv"
    summary_path = tmp_path / "case_audit_summary.json"
    markdown_path = tmp_path / "error_audit.md"
    assert csv_path.exists() and summary_path.exists() and markdown_path.exists()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        assert [row["case_id"] for row in csv.DictReader(handle)] == ["病例-甲", "病例-乙", "病例-丙"]
    assert json.loads(summary_path.read_text(encoding="utf-8"))["malignant"] == summary["malignant"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "来源" in markdown
    assert "无高置信度错分" in markdown
    assert "仅供人工核查、不证明标签错误或模态因果" in markdown
