"""对已有 flat5 检查点执行有限病例 SWE 稳定性审计。"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.datasets.dataset import MultiModalBreastDataset
from code.train.model_evidence_audit import (
    _load_json,
    _select_case_ids,
    compare_prediction_records,
)
from code.train.run_artifacts import save_json
from code.train.train_stage2 import (
    build_eval_loader,
    build_fair_splits,
    build_model,
    restore_manifest_splits,
)


CONDITIONS = (
    "intensity_0_9",
    "intensity_1_1",
    "horizontal_flip",
    "case_shuffle",
)
MALIGNANT_CLASS_COUNT = 4


def validate_flat5_checkpoint_args(saved_args: Mapping) -> None:
    """拒绝不是 flat5 的检查点，避免混用不同的类别空间。"""
    if not isinstance(saved_args, Mapping) or saved_args.get("task_mode") != "flat5":
        raise ValueError("SWE 审计仅支持 task_mode 为 flat5 的检查点")


def predict_malignant_logits(class_logits: torch.Tensor):
    """仅在 flat5 输出的前四个恶性类别内计算预测与置信度。"""
    if not isinstance(class_logits, torch.Tensor) or class_logits.ndim != 2:
        raise ValueError("class_logits 必须是二维张量")
    if class_logits.shape[1] < MALIGNANT_CLASS_COUNT:
        raise ValueError("class_logits 至少需要包含四个恶性类别")
    malignant_probabilities = torch.softmax(
        class_logits[:, :MALIGNANT_CLASS_COUNT], dim=1
    )
    confidence, predicted = malignant_probabilities.max(dim=1)
    return predicted, confidence


def apply_swe_perturbation(swe_tensor: torch.Tensor, condition: str) -> torch.Tensor:
    """对归一化 SWE 张量执行一个确定性扰动。"""
    if condition == "horizontal_flip":
        return torch.flip(swe_tensor, dims=(-1,))
    factors = {"intensity_0_9": 0.9, "intensity_1_1": 1.1}
    if condition not in factors:
        raise ValueError(f"未知 SWE 扰动条件：{condition}")
    pixels = ((swe_tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    return (pixels.mul(factors[condition]).clamp(0.0, 1.0) * 2.0) - 1.0


def build_shuffle_indices(case_ids: Sequence[str]) -> list[int]:
    """构造循环移动一位的确定性病例错配索引。"""
    if len(case_ids) < 2:
        raise ValueError("病例错配至少需要两个病例")
    return list(range(1, len(case_ids))) + [0]


class SWEPerturbationDataset(Dataset):
    """只修改 SWE、保留其他输入和标签不变的数据集包装器。"""

    def __init__(
        self,
        base_dataset: Dataset,
        condition: str,
        shuffle_indices: Sequence[int] | None = None,
    ):
        if condition not in CONDITIONS:
            raise ValueError(f"未知 SWE 扰动条件：{condition}")
        self.base_dataset = base_dataset
        self.condition = condition
        self.samples = base_dataset.samples
        self.shuffle_indices = None
        if condition == "case_shuffle":
            indices = (
                build_shuffle_indices([sample["case_id"] for sample in self.samples])
                if shuffle_indices is None
                else list(shuffle_indices)
            )
            if len(indices) != len(self.samples):
                raise ValueError("SWE 错配索引数量必须与病例数一致")
            if sorted(indices) != list(range(len(self.samples))):
                raise ValueError("SWE 错配索引必须是完整病例置换")
            if any(index == donor_index for index, donor_index in enumerate(indices)):
                raise ValueError("SWE 错配索引不能包含自配对")
            self.shuffle_indices = indices
        elif shuffle_indices is not None:
            raise ValueError("仅病例错配条件允许指定 SWE 错配索引")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.base_dataset[index])
        if self.condition == "case_shuffle":
            donor_index = self.shuffle_indices[index]
            donor = self.base_dataset[donor_index]
            item["swe_img"] = donor["swe_img"]
            item["donor_case_id"] = donor["case_id"]
            item["donor_label"] = int(donor["subtype_label"])
        else:
            item["swe_img"] = apply_swe_perturbation(
                item["swe_img"], self.condition
            )
            item["donor_case_id"] = item["case_id"]
            item["donor_label"] = int(item["subtype_label"])
        return item


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_stability(rows: Sequence[Mapping]) -> dict:
    """按扰动条件汇总预测改变、损伤、纠正和供体一致性。"""
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    summary = {}
    for condition, condition_rows in grouped.items():
        row_count = len(condition_rows)
        changed_count = sum(bool(row["prediction_changed"]) for row in condition_rows)
        base_correct_count = sum(bool(row["base_correct"]) for row in condition_rows)
        base_error_count = row_count - base_correct_count
        damaged_count = sum(
            bool(row["base_correct"]) and not bool(row["perturbed_correct"])
            for row in condition_rows
        )
        corrected_count = sum(
            not bool(row["base_correct"]) and bool(row["perturbed_correct"])
            for row in condition_rows
        )
        values = {
            "row_count": row_count,
            "prediction_changed_count": changed_count,
            "prediction_changed_rate": _safe_rate(changed_count, row_count),
            "mean_confidence_delta": float(
                np.mean([float(row["confidence_delta"]) for row in condition_rows])
            ),
            "base_correct_count": base_correct_count,
            "correct_damaged_count": damaged_count,
            "correct_damaged_rate": _safe_rate(damaged_count, base_correct_count),
            "base_error_count": base_error_count,
            "error_corrected_count": corrected_count,
            "error_corrected_rate": _safe_rate(corrected_count, base_error_count),
        }
        if condition == "case_shuffle":
            donor_agreement_count = sum(
                int(row["perturbed_predicted_label"]) == int(row["donor_label"])
                for row in condition_rows
            )
            values["donor_agreement_count"] = donor_agreement_count
            values["donor_agreement_rate"] = _safe_rate(
                donor_agreement_count, row_count
            )
        summary[condition] = values
    return summary


def _build_selected_dataset(dataset, case_ids: Sequence[str]):
    sample_by_id = {sample["case_id"]: sample for sample in dataset.samples}
    missing = [case_id for case_id in case_ids if case_id not in sample_by_id]
    if missing:
        raise ValueError(f"检查点测试集缺少选定病例：{missing[:5]}")
    return MultiModalBreastDataset(
        dataset.root_dir,
        split="test",
        img_size=dataset.img_size,
        max_text_len=dataset.max_text_len,
        samples=[sample_by_id[case_id] for case_id in case_ids],
        augment=False,
        tokenizer=dataset.tokenizer,
    )


def _predict_dataset(model, dataset, device, batch_size: int) -> list[dict]:
    """对一个确定性数据集执行预测并保留供体信息。"""
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


def validate_case_limits(max_error_cases: int, max_correct_cases: int) -> None:
    """确保 SWE 稳定性审计固定覆盖 24 个错误病例和 12 个正确病例。"""
    if max_error_cases != 24 or max_correct_cases != 12:
        raise ValueError("SWE 稳定性审计病例数必须固定为 24 个错误病例和 12 个正确病例")


def run_audit(args) -> dict:
    """加载已有检查点并执行固定条件的 SWE 稳定性审计。"""
    validate_case_limits(args.max_error_cases, args.max_correct_cases)
    if args.max_error_cases < 0 or args.max_correct_cases < 0:
        raise ValueError("病例数上限不能为负数")
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
    full_dataset = MultiModalBreastDataset(
        args.project_root,
        split="test",
        img_size=saved_args["img_size"],
        max_text_len=saved_args["max_text_len"],
        samples=splits["malignant_test"],
        augment=False,
    )
    case_ids = _select_case_ids(
        Path(args.case_audit_dir),
        Path(args.quality_flags),
        args.max_error_cases,
        args.max_correct_cases,
    )
    expected_cases = args.max_error_cases + args.max_correct_cases
    if len(case_ids) != expected_cases:
        raise RuntimeError(
            f"SWE 稳定性病例数异常：预期 {expected_cases}，实际 {len(case_ids)}"
        )
    selected_dataset = _build_selected_dataset(full_dataset, case_ids)
    base_rows = _predict_dataset(model, selected_dataset, device, args.batch_size)
    base_by_id = {row["case_id"]: row for row in base_rows}

    rows = []
    for condition in CONDITIONS:
        perturbed_rows = _predict_dataset(
            model,
            SWEPerturbationDataset(selected_dataset, condition),
            device,
            args.batch_size,
        )
        perturbed_by_id = {row["case_id"]: row for row in perturbed_rows}
        for change in compare_prediction_records(base_rows, perturbed_rows):
            base = base_by_id[change["case_id"]]
            perturbed = perturbed_by_id[change["case_id"]]
            perturbed_label = int(change["ablated_predicted_label"])
            true_label = int(base["true_label"])
            rows.append(
                {
                    "case_id": change["case_id"],
                    "condition": condition,
                    "true_label": true_label,
                    "base_predicted_label": int(change["base_predicted_label"]),
                    "perturbed_predicted_label": perturbed_label,
                    "base_confidence": float(change["base_confidence"]),
                    "perturbed_confidence": float(change["ablated_confidence"]),
                    "confidence_delta": float(change["confidence_delta"]),
                    "prediction_changed": bool(change["prediction_changed"]),
                    "base_correct": int(base["predicted_label"]) == true_label,
                    "perturbed_correct": perturbed_label == true_label,
                    "donor_case_id": perturbed["donor_case_id"],
                    "donor_label": int(perturbed["donor_label"]),
                }
            )

    expected_rows = len(case_ids) * len(CONDITIONS)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"SWE 稳定性记录数异常：预期 {expected_rows}，实际 {len(rows)}"
        )

    by_condition = summarize_stability(rows)
    summary = {
        "case_count": len(case_ids),
        "row_count": len(rows),
        "condition_count": len(CONDITIONS),
        "by_condition": by_condition,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "swe_stability_summary.json", summary)
    fieldnames = [
        "case_id",
        "condition",
        "true_label",
        "base_predicted_label",
        "perturbed_predicted_label",
        "base_confidence",
        "perturbed_confidence",
        "confidence_delta",
        "prediction_changed",
        "base_correct",
        "perturbed_correct",
        "donor_case_id",
        "donor_label",
    ]
    with (output_dir / "swe_stability_cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_lines = [
        "# SWE 稳定性审计报告",
        "",
        f"- 病例数：{len(case_ids)}",
        f"- 扰动条件数：{len(CONDITIONS)}",
        f"- 病例级扰动记录数：{len(rows)}",
        "",
        "| 条件 | 预测改变率 | 平均置信度变化 | 正确病例受损率 | 错误病例纠正率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        values = by_condition[condition]
        report_lines.append(
            f"| `{condition}` | {values['prediction_changed_rate']:.4f} | "
            f"{values['mean_confidence_delta']:.4f} | "
            f"{values['correct_damaged_rate']:.4f} | "
            f"{values['error_corrected_rate']:.4f} |"
        )
    shuffle_summary = by_condition["case_shuffle"]
    report_lines.extend(
        [
            "",
            "- 病例错配后的供体标签一致率："
            f"{shuffle_summary['donor_agreement_rate']:.4f}",
            "",
            "本报告只描述模型对 SWE 输入扰动的敏感性，不等同于真实病灶定位或因果证据。",
            "",
        ]
    )
    (output_dir / "swe_stability_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构造 SWE 稳定性审计命令行解析器。"""
    parser = argparse.ArgumentParser(description="已有检查点 SWE 稳定性审计")
    parser.add_argument("--project_root", required=True, help="包含 data 目录的项目根目录")
    parser.add_argument("--run_dir", required=True, help="已有 flat5 运行目录")
    parser.add_argument("--case_audit_dir", required=True, help="病例审计目录")
    parser.add_argument("--quality_flags", required=True, help="数据质量标记 CSV")
    parser.add_argument("--output_dir", required=True, help="SWE 稳定性审计输出目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--max_error_cases", type=int, default=24, help="最多错误病例数")
    parser.add_argument("--max_correct_cases", type=int, default=12, help="最多正确病例数")
    parser.add_argument("--batch_size", type=int, default=8, help="推理批量大小")
    return parser


if __name__ == "__main__":
    run_audit(build_parser().parse_args())
