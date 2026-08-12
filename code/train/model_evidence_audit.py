"""对已有 flat5 检查点执行有限病例模型证据审计。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.datasets.dataset import MultiModalBreastDataset
from code.train.run_artifacts import save_json
from code.train.train_stage2 import (
    build_eval_loader,
    build_fair_splits,
    build_model,
    restore_manifest_splits,
)


def mask_text_terms(raw_text: str, terms: Sequence[str]) -> str:
    """用统一遮蔽标记替换指定文本关键词。"""
    result = "" if raw_text is None else str(raw_text)
    for term in terms:
        if term:
            result = result.replace(term, "[遮蔽]")
    return result


def compare_prediction_records(
    base_rows: Sequence[Mapping],
    ablated_rows: Sequence[Mapping],
) -> list[dict]:
    """按病例编号对齐完整输入与模态屏蔽预测。"""
    if len(base_rows) != len(ablated_rows):
        raise ValueError("两组预测记录的病例数量不一致")
    results = []
    for base, ablated in zip(base_rows, ablated_rows):
        if base.get("case_id") != ablated.get("case_id"):
            raise ValueError("两组预测记录的病例编号不一致")
        base_confidence = float(base["confidence"])
        ablated_confidence = float(ablated["confidence"])
        results.append(
            {
                "case_id": base["case_id"],
                "base_predicted_label": int(base["predicted_label"]),
                "ablated_predicted_label": int(ablated["predicted_label"]),
                "base_confidence": base_confidence,
                "ablated_confidence": ablated_confidence,
                "confidence_delta": ablated_confidence - base_confidence,
                "prediction_changed": (
                    int(base["predicted_label"]) != int(ablated["predicted_label"])
                ),
            }
        )
    return results


def compute_spatial_concentration(
    saliency: np.ndarray,
    center_fraction: float = 0.6,
) -> dict:
    """计算中心区域、边缘区域和总显著性比例。"""
    values = np.asarray(saliency, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("显著性图必须是有限的二维数组")
    if not 0 < center_fraction <= 1:
        raise ValueError("中心区域比例必须在 0 和 1 之间")
    values = np.abs(values)
    total = float(values.sum())
    height, width = values.shape
    center_height = max(1, int(round(height * center_fraction)))
    center_width = max(1, int(round(width * center_fraction)))
    top = (height - center_height) // 2
    left = (width - center_width) // 2
    center = values[top:top + center_height, left:left + center_width]
    center_sum = float(center.sum())
    center_ratio = center_sum / total if total else 0.0
    return {
        "center_ratio": center_ratio,
        "edge_ratio": 1.0 - center_ratio,
        "total_saliency": total,
    }


def summarize_evidence(rows: Sequence[Mapping]) -> dict:
    """按模态和错误类型汇总预测变化。"""
    by_modality = defaultdict(lambda: {"row_count": 0, "changed_prediction_count": 0})
    error_counts = Counter()
    changed = 0
    for row in rows:
        modality = str(row.get("modality", "未知"))
        by_modality[modality]["row_count"] += 1
        if row.get("prediction_changed"):
            by_modality[modality]["changed_prediction_count"] += 1
            changed += 1
        if row.get("error_type") is not None:
            error_counts[str(row["error_type"])] += 1
    return {
        "row_count": len(rows),
        "changed_prediction_count": changed,
        "by_modality": dict(by_modality),
        "error_type_counts": dict(error_counts),
    }


def summarize_saliency(rows: Sequence[Mapping]) -> dict:
    """按模态汇总显著性空间集中度，不保存显著性张量。"""
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["modality"])].append(row)
    summary = {}
    for modality, modality_rows in grouped.items():
        summary[modality] = {
            "row_count": len(modality_rows),
            "mean_center_ratio": float(
                np.mean([float(row["center_ratio"]) for row in modality_rows])
            ),
            "mean_edge_ratio": float(
                np.mean([float(row["edge_ratio"]) for row in modality_rows])
            ),
        }
    return summary


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 JSON：{path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"JSON 必须是对象：{path}")
    return data


def _load_raw_text(project_root: Path, case_id: str) -> str:
    path = project_root / "data" / "texts" / f"{case_id}.json"
    if not path.is_file():
        return ""
    try:
        data = _load_json(path)
    except ValueError:
        return ""
    return str(data.get("raw_text", ""))


def _select_case_ids(case_audit_dir: Path, quality_flags: Path, max_error: int, max_correct: int) -> list[str]:
    """从已有病例审计和质量标记中确定性选择有限病例。"""
    prediction_path = Path(case_audit_dir) / "case_predictions.csv"
    rows = list(csv.DictReader(prediction_path.open(encoding="utf-8-sig")))
    flags = {
        row["case_id"]: row
        for row in csv.DictReader(Path(quality_flags).open(encoding="utf-8-sig"))
    }
    errors = [row["case_id"] for row in rows if row.get("error_type") != "正确"]
    correct = [row["case_id"] for row in rows if row.get("error_type") == "正确"]
    selected = errors[:max_error] + correct[:max_correct]
    return [case_id for case_id in selected if case_id in flags]


def _predict(model, dataset, device, case_ids, ablate_modalities):
    """对有限病例执行一次确定性预测，不保存原始输入。"""
    sample_by_id = {sample["case_id"]: sample for sample in dataset.samples}
    missing = [case_id for case_id in case_ids if case_id not in sample_by_id]
    if missing:
        raise ValueError(f"检查点测试集缺少选定病例：{missing[:5]}")
    samples = [sample_by_id[case_id] for case_id in case_ids]
    subset = MultiModalBreastDataset(
        dataset.root_dir,
        split="test",
        img_size=dataset.img_size,
        max_text_len=dataset.max_text_len,
        samples=samples,
        augment=False,
        ablate_modalities=ablate_modalities,
    )
    loader = build_eval_loader(subset, argparse.Namespace(batch_size=16, num_workers=0))
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["bus_img"].to(device), batch["swe_img"].to(device),
                batch["cdfi_img"].to(device), batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            probabilities = torch.softmax(outputs["class_logits"], dim=1)
            confidence, predicted = probabilities.max(dim=1)
            for case_id, label, pred, conf in zip(
                batch["case_id"],
                batch["subtype_label"].tolist(), predicted.tolist(), confidence.tolist(),
            ):
                rows.append({
                    "case_id": case_id,
                    "true_label": int(label),
                    "predicted_label": int(pred),
                    "confidence": float(conf),
                })
    return rows


def _save_saliency(model, dataset, device, case_ids, output_dir, max_maps):
    """为有限病例保存归一化显著性图及空间集中度，不保存梯度张量。"""
    output_dir = Path(output_dir)
    maps_dir = output_dir / "saliency_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # 每个病例生成三张图，因此 max_maps 表示 PNG 总数而不是病例数。
    max_case_maps = max_maps // 3
    for case_id in case_ids[:max_case_maps]:
        sample = next(sample for sample in dataset.samples if sample["case_id"] == case_id)
        item = dataset[dataset.samples.index(sample)]
        inputs = {key: value.unsqueeze(0).to(device) for key, value in item.items() if key.endswith("_img")}
        bus = inputs["bus_img"].requires_grad_(True)
        swe = inputs["swe_img"].requires_grad_(True)
        cdfi = inputs["cdfi_img"].requires_grad_(True)
        model.zero_grad(set_to_none=True)
        outputs = model(bus, swe, cdfi, item["input_ids"].unsqueeze(0).to(device), item["attention_mask"].unsqueeze(0).to(device))
        predicted = outputs["class_logits"].argmax(dim=1)
        outputs["class_logits"][0, predicted].backward()
        for name, tensor in (("bus", bus), ("swe", swe), ("cdfi", cdfi)):
            saliency = tensor.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
            stats = compute_spatial_concentration(saliency)
            image = (saliency / max(float(saliency.max()), 1e-12) * 255).astype(np.uint8)
            Image.fromarray(image).save(maps_dir / f"{case_id}-{name}.png")
            rows.append({"case_id": case_id, "modality": name, **stats})
    return rows


def run_audit(args) -> dict:
    """加载既有检查点并执行有限病例证据审计。"""
    if args.max_error_cases < 0 or args.max_correct_cases < 0 or args.max_maps < 0:
        raise ValueError("病例数和显著性图上限不能为负数")
    device = torch.device(args.device)
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
    run_dir = Path(args.run_dir)
    saved_args = _load_json(run_dir / "args.json")
    manifest = _load_json(run_dir / "split_manifest.json")
    checkpoint = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=False)
    model = build_model(argparse.Namespace(**saved_args), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    splits = restore_manifest_splits(
        build_fair_splits(args.project_root, "flat5", saved_args.get("malignant_metadata", "metadata.csv"), saved_args.get("benign_metadata", "metadata_5class.csv")),
        manifest,
    )
    dataset = MultiModalBreastDataset(args.project_root, split="test", img_size=saved_args["img_size"], max_text_len=saved_args["max_text_len"], samples=splits["malignant_test"], augment=False)
    case_ids = _select_case_ids(Path(args.case_audit_dir), Path(args.quality_flags), args.max_error_cases, args.max_correct_cases)
    base = _predict(model, dataset, device, case_ids, [])
    rows = []
    for modality in ("swe", "cdfi", "text"):
        ablated = _predict(model, dataset, device, case_ids, [modality])
        changes = compare_prediction_records(base, ablated)
        for change in changes:
            change["modality"] = modality
            source = next(item for item in base if item["case_id"] == change["case_id"])
            change["true_label"] = source["true_label"]
            change["error_type"] = "正确" if source["predicted_label"] == source["true_label"] else "错误"
            rows.append(change)
    saliency_rows = _save_saliency(model, dataset, device, case_ids, args.output_dir, args.max_maps)
    summary = summarize_evidence(rows)
    summary["saliency_row_count"] = len(saliency_rows)
    summary["saliency_by_modality"] = summarize_saliency(saliency_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "model_evidence_summary.json", summary)
    with (output_dir / "model_evidence_cases.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "saliency_statistics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = [
            "case_id",
            "modality",
            "center_ratio",
            "edge_ratio",
            "total_saliency",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(saliency_rows)
    (output_dir / "model_evidence_report.md").write_text(
        "# 模型证据审计报告\n\n"
        f"- 重点病例数：{len(case_ids)}\n"
        f"- 模态扰动记录数：{len(rows)}\n"
        f"- 显著性统计记录数：{len(saliency_rows)}\n\n"
        "空间集中度只表示模型输入梯度的空间分布，不等同于真实病灶定位。\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构造模型证据审计命令行解析器。"""
    parser = argparse.ArgumentParser(description="已有检查点模型证据审计")
    parser.add_argument("--project_root", required=True, help="包含 data 目录的项目根目录")
    parser.add_argument("--run_dir", required=True, help="已有 flat5 运行目录")
    parser.add_argument("--case_audit_dir", required=True, help="病例审计目录")
    parser.add_argument("--quality_flags", required=True, help="数据质量标记 CSV")
    parser.add_argument("--output_dir", required=True, help="证据审计输出目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--max_error_cases", type=int, default=24, help="最多错误病例数")
    parser.add_argument("--max_correct_cases", type=int, default=12, help="最多正确病例数")
    parser.add_argument("--max_maps", type=int, default=12, help="最多保存的显著性 PNG 总数")
    return parser


if __name__ == "__main__":
    run_audit(build_parser().parse_args())
