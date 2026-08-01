"""生成 flat5 恶性病例的可序列化错误审计产物。"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from code.train.run_artifacts import save_json
from code.utils.evaluation import SUBTYPE_NAMES, evaluate_predictions


FIVE_CLASS_NAMES = [*SUBTYPE_NAMES, "良性"]
_MODALITY_FIELDS = ("bus_exists", "swe_exists", "cdfi_exists", "text_exists")
_METRIC_TOLERANCE = 1e-12


def build_case_records(case_ids, subtype_labels, class_logits, modality_exists):
    """从 flat5 logits 构造不含原始输入内容的病例审计记录。"""
    case_ids = _validate_case_ids(case_ids)
    labels = _validate_subtype_labels(subtype_labels, len(case_ids))
    logits = _validate_logits(class_logits, len(case_ids))
    modality_rows = _validate_modalities(modality_exists, len(case_ids))
    five_probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    conditional_probabilities = torch.softmax(logits[:, :4], dim=1).detach().cpu().numpy()
    conditional_predictions = logits[:, :4].argmax(dim=1).detach().cpu().numpy()
    end_to_end_predictions = logits.argmax(dim=1).detach().cpu().numpy()

    records = []
    for index, case_id in enumerate(case_ids):
        true_label = int(labels[index])
        conditional_label = int(conditional_predictions[index])
        end_to_end_label = int(end_to_end_predictions[index])
        conditional_correct = conditional_label == true_label
        end_to_end_correct = end_to_end_label == true_label
        error_type = _error_type(conditional_correct, end_to_end_label)
        records.append(
            {
                "case_id": case_id,
                "true_subtype_label": true_label,
                "true_subtype_name": SUBTYPE_NAMES[true_label],
                "conditional_predicted_subtype_label": conditional_label,
                "conditional_predicted_subtype_name": SUBTYPE_NAMES[conditional_label],
                "end_to_end_predicted_class_label": end_to_end_label,
                "end_to_end_predicted_class_name": FIVE_CLASS_NAMES[end_to_end_label],
                "five_class_probabilities": [float(value) for value in five_probabilities[index]],
                "conditional_probabilities": [
                    float(value) for value in conditional_probabilities[index]
                ],
                "conditional_confidence": float(conditional_probabilities[index].max()),
                "is_conditional_correct": conditional_correct,
                "is_end_to_end_correct": end_to_end_correct,
                "error_type": error_type,
                **modality_rows[index],
            }
        )
    return records


def build_audit_summary(records, saved_metrics):
    """复现并校验保存的恶性条件四分类指标，汇总病例错误。"""
    if not isinstance(records, list) or not records:
        raise ValueError("records 必须是非空病例记录列表")
    if not isinstance(saved_metrics, dict) or not isinstance(saved_metrics.get("malignant"), dict):
        raise ValueError("saved_metrics 必须包含 malignant 指标区域")
    try:
        y_true = [record["true_subtype_label"] for record in records]
        y_pred = [record["conditional_predicted_subtype_label"] for record in records]
    except (KeyError, TypeError) as error:
        raise ValueError("records 缺少条件四分类指标所需字段") from error
    malignant = evaluate_predictions(y_true, y_pred, SUBTYPE_NAMES)
    _assert_reproducible(saved_metrics["malignant"], malignant)
    error_counts = {"正确": 0, "亚型错分": 0, "恶性病例预测为良性": 0}
    for record in records:
        error_type = record.get("error_type")
        if error_type not in error_counts:
            raise ValueError("records 包含未知错误类型")
        error_counts[error_type] += 1
    return {
        "source": "flat5 恶性病例条件四分类预测",
        "record_count": len(records),
        "malignant": malignant,
        "error_type_distribution": error_counts,
    }


def validate_output_directory(output_dir, overwrite):
    """创建空输出目录，并拒绝未经确认的非空目录。"""
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("输出路径必须是目录")
    output_path.mkdir(parents=True, exist_ok=True)
    if any(output_path.iterdir()) and not overwrite:
        raise ValueError("输出目录非空，必须显式允许覆盖")
    return output_path


def write_audit_outputs(output_dir, records, summary):
    """写入 CSV、JSON 和供人工核查的 Markdown 审计报告。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not isinstance(records, list) or not records:
        raise ValueError("records 必须是非空病例记录列表")
    if not isinstance(summary, dict):
        raise ValueError("summary 必须是字典")
    _write_csv(output_path / "case_predictions.csv", records)
    save_json(output_path / "case_audit_summary.json", summary)
    (output_path / "error_audit.md").write_text(
        _render_markdown(records, summary), encoding="utf-8"
    )


def _validate_case_ids(case_ids):
    """校验病例编号非空、可审计且不重复。"""
    if isinstance(case_ids, (str, bytes)):
        raise ValueError("case_ids 必须是非空病例编号序列")
    try:
        values = list(case_ids)
    except TypeError as error:
        raise ValueError("case_ids 必须是非空病例编号序列") from error
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("case_ids 必须包含非空病例编号")
    if len(set(values)) != len(values):
        raise ValueError("case_ids 不允许重复")
    return values


def _validate_subtype_labels(subtype_labels, expected_length):
    """校验恶性亚型标签与病例数量一致。"""
    labels = np.asarray(subtype_labels)
    if labels.ndim != 1 or len(labels) != expected_length:
        raise ValueError("subtype_labels 必须是一维且与病例数量一致")
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= 4)):
        raise ValueError("subtype_labels 必须是 0 到 3 的整数标签")
    return labels


def _validate_logits(class_logits, expected_length):
    """校验仅支持有限的 [N, 5] flat5 logits。"""
    if not isinstance(class_logits, torch.Tensor) or class_logits.ndim != 2:
        raise ValueError("class_logits 必须是二维张量")
    if tuple(class_logits.shape) != (expected_length, 5):
        raise ValueError("class_logits 必须具有 [N, 5] 形状且与病例数量一致")
    if not torch.isfinite(class_logits).all():
        raise ValueError("class_logits 必须全部为有限数")
    return class_logits.detach().to(dtype=torch.float64, device="cpu")


def _validate_modalities(modality_exists, expected_length):
    """将每例四种模态存在性规范为具名布尔字段。"""
    if isinstance(modality_exists, (str, bytes)):
        raise ValueError("modality_exists 必须与病例数量一致")
    try:
        rows = list(modality_exists)
    except TypeError as error:
        raise ValueError("modality_exists 必须与病例数量一致") from error
    if len(rows) != expected_length:
        raise ValueError("modality_exists 必须与病例数量一致")
    normalized = []
    for row in rows:
        if isinstance(row, dict):
            if set(row) != set(_MODALITY_FIELDS):
                raise ValueError("每例模态存在性必须恰含四种模态")
            values = [row[field] for field in _MODALITY_FIELDS]
        else:
            if isinstance(row, (str, bytes)):
                raise ValueError("每例模态存在性必须包含四项")
            try:
                values = list(row)
            except TypeError as error:
                raise ValueError("每例模态存在性必须包含四项") from error
        if len(values) != 4 or any(not isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("每例模态存在性必须是四个布尔值")
        normalized.append(dict(zip(_MODALITY_FIELDS, (bool(value) for value in values))))
    return normalized


def _error_type(conditional_correct, end_to_end_label):
    """按端到端良性优先的规则标注病例错误类型。"""
    if end_to_end_label == 4:
        return "恶性病例预测为良性"
    return "正确" if conditional_correct else "亚型错分"


def _assert_reproducible(saved, actual, path="malignant"):
    """以严格容差递归确认保存指标可由记录精确复现。"""
    if not saved:
        raise ValueError("无法复现：保存的 malignant 指标区域不能为空")
    if isinstance(saved, dict):
        if not isinstance(actual, dict) or set(saved) != set(actual):
            raise ValueError(f"无法复现：{path} 的指标键不一致")
        for key in saved:
            _assert_reproducible(saved[key], actual[key], f"{path}.{key}")
        return
    if isinstance(saved, list):
        if not isinstance(actual, list) or len(saved) != len(actual):
            raise ValueError(f"无法复现：{path} 的长度不一致")
        for index, value in enumerate(saved):
            _assert_reproducible(value, actual[index], f"{path}[{index}]")
        return
    if isinstance(saved, (bool, np.bool_)) or not isinstance(saved, (int, float, np.number)):
        raise ValueError(f"无法复现：{path} 不是有效指标")
    if not np.isfinite(saved) or not np.isfinite(actual) or abs(float(saved) - float(actual)) > _METRIC_TOLERANCE:
        raise ValueError(f"无法复现：{path} 与病例记录不一致")


def _write_csv(path, records):
    """按原始记录顺序写入便于筛选的病例预测 CSV。"""
    fieldnames = [
        "case_id", "true_subtype_label", "true_subtype_name",
        "conditional_predicted_subtype_label", "conditional_predicted_subtype_name",
        "end_to_end_predicted_class_label", "end_to_end_predicted_class_name",
        "conditional_confidence", "is_conditional_correct", "is_end_to_end_correct",
        "error_type", *_MODALITY_FIELDS,
        *[f"five_class_probability_{index}" for index in range(5)],
        *[f"conditional_probability_{index}" for index in range(4)],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {field: record[field] for field in fieldnames if field in record}
            row.update({
                f"five_class_probability_{index}": record["five_class_probabilities"][index]
                for index in range(5)
            })
            row.update({
                f"conditional_probability_{index}": record["conditional_probabilities"][index]
                for index in range(4)
            })
            writer.writerow(row)


def _render_markdown(records, summary):
    """渲染覆盖错误病例与高置信度分支的人工核查报告。"""
    malignant = summary["malignant"]
    lines = [
        "# 病例级错误审计",
        "",
        f"来源：{summary['source']}。共纳入 {summary['record_count']} 例恶性病例。",
        "",
        "## 各真实亚型",
        "",
        "|真实亚型|总数|正确数|错误数|召回率|",
        "|---|---:|---:|---:|---:|",
    ]
    for index, name in enumerate(SUBTYPE_NAMES):
        matching = [record for record in records if record["true_subtype_label"] == index]
        correct = sum(record["is_conditional_correct"] for record in matching)
        total = len(matching)
        lines.append(f"|{name}|{total}|{correct}|{total - correct}|{malignant['per_class'][name]['recall']:.6f}|")
    lines.extend(["", "## 混淆矩阵", "", "真实类别为行，条件预测类别为列。", ""])
    lines.extend(["|真实/预测|" + "|".join(SUBTYPE_NAMES) + "|", "|---|---:|---:|---:|---:|"])
    for index, name in enumerate(SUBTYPE_NAMES):
        lines.append("|" + name + "|" + "|".join(str(value) for value in malignant["confusion_matrix"][index]) + "|")
    errors = [record for record in records if record["error_type"] != "正确"]
    lines.extend(["", "## 全部错误病例", ""])
    lines.extend(_render_error_rows(errors))
    high_confidence = [record for record in errors if record["conditional_confidence"] >= 0.80]
    lines.extend(["", "## 高置信度错分", "", "条件置信度阈值：0.80。", ""])
    if high_confidence:
        lines.extend(_render_error_rows(high_confidence))
    else:
        lines.append("无高置信度错分。")
    lines.extend(["", "## 限制", "", "仅供人工核查、不证明标签错误或模态因果。", ""])
    return "\n".join(lines)


def _render_error_rows(records):
    """仅渲染允许出现在审计报告中的病例级字段。"""
    if not records:
        return ["无错误病例。"]
    lines = ["|病例 ID|真实亚型|条件预测|端到端预测|条件置信度|错误类型|", "|---|---|---|---|---:|---|"]
    for record in records:
        lines.append(
            f"|{record['case_id']}|{record['true_subtype_name']}|"
            f"{record['conditional_predicted_subtype_name']}|"
            f"{record['end_to_end_predicted_class_name']}|"
            f"{record['conditional_confidence']:.6f}|{record['error_type']}|"
        )
    return lines
