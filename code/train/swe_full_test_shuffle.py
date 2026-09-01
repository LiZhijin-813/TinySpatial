"""对完整恶性测试集执行原始输入与 SWE 病例循环错配对照。"""

from __future__ import annotations

import argparse
import csv
import sys
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.datasets.dataset import MultiModalBreastDataset
from code.train.model_evidence_audit import (
    _load_json,
    compare_prediction_records,
)
from code.train.run_artifacts import save_json
from code.train.swe_stability_audit import (
    SWEPerturbationDataset,
    predict_malignant_logits,
    validate_flat5_checkpoint_args,
)
from code.train.train_stage2 import (
    build_eval_loader,
    build_fair_splits,
    build_model,
    restore_manifest_splits,
)
from code.utils.evaluation import evaluate_predictions


CLASS_NAMES = ["Luminal A", "Luminal B", "HER2+", "TNBC"]


def build_full_shuffle_indices(case_ids: Sequence[str]) -> list[int]:
    """构造不含自配对的确定性循环错配索引。"""
    if len(case_ids) < 2:
        raise ValueError("病例错配至少需要两个病例")
    return list(range(1, len(case_ids))) + [0]


def build_case_comparison_rows(
    base_rows: Sequence[Mapping],
    shuffled_rows: Sequence[Mapping],
    expected_donor_by_case: Mapping | None = None,
    true_label_by_case: Mapping | None = None,
) -> list[dict]:
    """校验病例对齐和供体信息，并生成原始与错配对照记录。"""
    if len(base_rows) != len(shuffled_rows):
        raise ValueError("两组预测记录的病例数量不一致")
    base_case_ids = [row.get("case_id") for row in base_rows]
    shuffled_case_ids = [row.get("case_id") for row in shuffled_rows]
    if len(set(base_case_ids)) != len(base_case_ids) or len(
        set(shuffled_case_ids)
    ) != len(shuffled_case_ids):
        raise ValueError("病例编号必须唯一")
    if set(base_case_ids) != set(shuffled_case_ids):
        raise ValueError("病例编号集合不一致")
    if (expected_donor_by_case is None) != (true_label_by_case is None):
        raise ValueError("供体映射与真实标签映射必须同时提供")
    if expected_donor_by_case is not None:
        case_id_set = set(base_case_ids)
        if set(expected_donor_by_case) != case_id_set:
            raise ValueError("供体映射病例编号集合不一致")
        if set(true_label_by_case) != case_id_set:
            raise ValueError("真实标签映射病例编号集合不一致")
        expected_donor_ids = list(expected_donor_by_case.values())
        if set(expected_donor_ids) != case_id_set or len(
            set(expected_donor_ids)
        ) != len(expected_donor_ids):
            raise ValueError("供体映射必须覆盖测试集内每个病例且恰好一次")
    shuffled_by_id = {
        row["case_id"]: row for row in shuffled_rows
    }
    aligned_shuffled_rows = [shuffled_by_id[case_id] for case_id in base_case_ids]
    changes = compare_prediction_records(base_rows, aligned_shuffled_rows)
    rows = []
    for change, base, shuffled in zip(
        changes, base_rows, aligned_shuffled_rows
    ):
        if base["true_label"] != shuffled["true_label"]:
            raise ValueError(f"病例 {base['case_id']} 的真实标签不一致")
        donor_case_id = shuffled.get("donor_case_id")
        if donor_case_id == shuffled["case_id"]:
            raise ValueError(f"病例 {shuffled['case_id']} 的供体病例不能与当前病例相同")
        donor_label = shuffled.get("donor_label")
        if isinstance(donor_label, bool) or not isinstance(donor_label, Integral):
            raise ValueError("供体标签必须是整数")
        if not 0 <= int(donor_label) < len(CLASS_NAMES):
            raise ValueError("供体标签必须是四分类整数")
        if expected_donor_by_case is not None:
            if donor_case_id not in expected_donor_by_case:
                raise ValueError("供体病例不属于测试集")
            if donor_case_id != expected_donor_by_case[shuffled["case_id"]]:
                raise ValueError("供体映射不符合固定循环")
            if int(donor_label) != int(true_label_by_case[donor_case_id]):
                raise ValueError("供体标签与供体病例真实标签不一致")
        rows.append(
            {
                "case_id": change["case_id"],
                "true_label": int(base["true_label"]),
                "base_predicted_label": int(change["base_predicted_label"]),
                "shuffled_predicted_label": int(change["ablated_predicted_label"]),
                "base_confidence": float(change["base_confidence"]),
                "shuffled_confidence": float(change["ablated_confidence"]),
                "confidence_delta": float(change["confidence_delta"]),
                "prediction_changed": bool(change["prediction_changed"]),
                "donor_case_id": donor_case_id,
                "donor_label": int(donor_label),
            }
        )
    return rows


def _metrics_for_rows(rows: Sequence[Mapping], prediction_key: str) -> dict:
    result = evaluate_predictions(
        [int(row["true_label"]) for row in rows],
        [int(row[prediction_key]) for row in rows],
        class_names=CLASS_NAMES,
    )
    return {
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "balanced_accuracy": result["balanced_accuracy"],
        "per_class_f1": [result["per_class"][name]["f1"] for name in CLASS_NAMES],
        "confusion_matrix": result["confusion_matrix"],
        "prediction_distribution": result["prediction_distribution"],
    }


def summarize_condition_metrics(rows: Sequence[Mapping]) -> dict:
    """汇总原始输入和 SWE 错配输入的四分类指标。"""
    if not rows:
        raise ValueError("指标汇总至少需要一条病例记录")
    return {
        "base": _metrics_for_rows(rows, "base_predicted_label"),
        "shuffled": _metrics_for_rows(rows, "shuffled_predicted_label"),
    }


def _validate_device(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求的 CUDA 设备不可用")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"指定的 CUDA 设备编号超出范围：{device.index}，"
            f"可用设备数为 {torch.cuda.device_count()}"
        )


def _predict_dataset(model, dataset, device, batch_size: int) -> list[dict]:
    """对数据集执行确定性预测并保留病例与供体信息。"""
    loader = build_eval_loader(
        dataset,
        argparse.Namespace(batch_size=batch_size, num_workers=0),
    )
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["bus_img"].to(device),
                batch["swe_img"].to(device),
                batch["cdfi_img"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            predicted, confidence = predict_malignant_logits(outputs["class_logits"])
            donor_case_ids = batch.get("donor_case_id", batch["case_id"])
            donor_labels = batch.get("donor_label", batch["subtype_label"])
            for case_id, label, pred, conf, donor_id, donor_label in zip(
                batch["case_id"],
                batch["subtype_label"].tolist(),
                predicted.tolist(),
                confidence.tolist(),
                donor_case_ids,
                donor_labels.tolist(),
            ):
                rows.append(
                    {
                        "case_id": case_id,
                        "true_label": int(label),
                        "predicted_label": int(pred),
                        "confidence": float(conf),
                        "donor_case_id": donor_id,
                        "donor_label": int(donor_label),
                    }
                )
    return rows


def run_audit(args) -> dict:
    """加载检查点并执行完整恶性测试集的 SWE 循环错配审计。"""
    if args.batch_size <= 0:
        raise ValueError("批量大小必须为正数")
    device = torch.device(args.device)
    _validate_device(device)
    run_dir = Path(args.run_dir)
    saved_args = _load_json(run_dir / "args.json")
    validate_flat5_checkpoint_args(saved_args)
    manifest = _load_json(run_dir / "split_manifest.json")
    checkpoint = torch.load(
        run_dir / "best_model.pth", map_location=device, weights_only=False
    )
    model = build_model(argparse.Namespace(**saved_args), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    splits = restore_manifest_splits(
        build_fair_splits(
            args.project_root,
            "flat5",
            saved_args.get("malignant_metadata", "metadata.csv"),
            saved_args.get("benign_metadata", "metadata_5class.csv"),
        ),
        manifest,
    )
    malignant_test = splits["malignant_test"]
    if len(malignant_test) != 82:
        raise RuntimeError(f"完整恶性测试集病例数异常：预期 82，实际 {len(malignant_test)}")
    case_ids = [sample["case_id"] for sample in malignant_test]
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("完整恶性测试集病例编号必须唯一")
    shuffle_indices = build_full_shuffle_indices(case_ids)
    expected_donor_by_case = {
        case_id: case_ids[shuffle_indices[index]]
        for index, case_id in enumerate(case_ids)
    }
    true_label_by_case = {
        sample["case_id"]: int(sample["subtype_label"])
        for sample in malignant_test
    }
    dataset = MultiModalBreastDataset(
        args.project_root,
        split="test",
        img_size=saved_args["img_size"],
        max_text_len=saved_args["max_text_len"],
        samples=malignant_test,
        augment=False,
    )
    base_rows = _predict_dataset(model, dataset, device, args.batch_size)
    shuffled_rows = _predict_dataset(
        model,
        SWEPerturbationDataset(
            dataset, "case_shuffle", shuffle_indices=shuffle_indices
        ),
        device,
        args.batch_size,
    )
    if len(base_rows) != 82 or len(shuffled_rows) != 82:
        raise RuntimeError(
            f"SWE 全测试集记录数异常：原始 {len(base_rows)}，错配 {len(shuffled_rows)}，预期均为 82"
        )
    rows = build_case_comparison_rows(
        base_rows,
        shuffled_rows,
        expected_donor_by_case=expected_donor_by_case,
        true_label_by_case=true_label_by_case,
    )
    if len(rows) != 82:
        raise RuntimeError(f"SWE 全测试集对齐记录数异常：预期 82，实际 {len(rows)}")
    metrics = summarize_condition_metrics(rows)
    changed_count = sum(row["prediction_changed"] for row in rows)
    summary = {
        "case_count": len(rows),
        "prediction_changed_count": int(changed_count),
        "prediction_changed_rate": float(changed_count / len(rows)),
        "metrics": metrics,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "swe_full_test_summary.json", summary)
    fieldnames = list(rows[0].keys())
    with (output_dir / "swe_full_test_cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report_lines = [
        "# 完整恶性测试集 SWE 病例错配审计报告",
        "",
        f"- 病例数：{len(rows)}",
        f"- 预测改变数：{changed_count}",
        f"- 预测改变率：{changed_count / len(rows):.4f}",
        "",
        "| 输入条件 | 准确率 | Macro-F1 | 平衡准确率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, label in (("base", "原始输入"), ("shuffled", "SWE 循环错配")):
        values = metrics[name]
        report_lines.append(
            f"| {label} | {values['accuracy']:.4f} | "
            f"{values['macro_f1']:.4f} | {values['balanced_accuracy']:.4f} |"
        )
        report_lines.extend(
            [
                "",
                f"- {label}逐类 F1（Luminal A、Luminal B、HER2+、TNBC）："
                + "、".join(f"{value:.4f}" for value in values["per_class_f1"]),
                f"- {label}混淆矩阵："
                + "；".join(
                    "[" + ", ".join(str(value) for value in row) + "]"
                    for row in values["confusion_matrix"]
                ),
                f"- {label}预测分布（Luminal A、Luminal B、HER2+、TNBC）："
                + "、".join(str(value) for value in values["prediction_distribution"]),
            ]
        )
    (output_dir / "swe_full_test_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构造完整测试集 SWE 审计命令行解析器。"""
    parser = argparse.ArgumentParser(description="完整恶性测试集 SWE 循环错配审计")
    parser.add_argument("--project_root", required=True, help="包含 data 目录的项目根目录")
    parser.add_argument("--run_dir", required=True, help="已有运行目录")
    parser.add_argument("--output_dir", required=True, help="审计输出目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--batch_size", type=int, default=8, help="推理批量大小")
    return parser


if __name__ == "__main__":
    run_audit(build_parser().parse_args())
