"""固定检查点，仅在恶性验证集比较真实与循环错配 SWE。"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.datasets.dataset import MultiModalBreastDataset
from code.train.model_evidence_audit import _load_json
from code.train.run_artifacts import save_json
from code.train.swe_full_test_shuffle import (
    _predict_dataset,
    _validate_device,
    build_case_comparison_rows,
    summarize_condition_metrics,
)
from code.train.swe_stability_audit import (
    SWEPerturbationDataset,
    validate_flat5_checkpoint_args,
)
from code.train.train_stage2 import (
    build_fair_splits,
    build_model,
    restore_manifest_splits,
)


def prepare_validation_cases(splits):
    """只选恶性验证病例并构造跨亚型的一对一错配。"""
    samples = splits["malignant_val"]
    case_ids = [sample["case_id"] for sample in samples]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("恶性验证集病例编号必须唯一")
    labels = [int(sample["subtype_label"]) for sample in samples]
    max_class_count = max(Counter(labels).values())
    if max_class_count * 2 > len(samples):
        raise ValueError("无法为所有病例找到不同亚型供体")
    ordered_indices = sorted(
        range(len(samples)), key=lambda index: (labels[index], case_ids[index])
    )
    shuffle_indices = [0] * len(samples)
    for position, source_index in enumerate(ordered_indices):
        shuffle_indices[source_index] = ordered_indices[
            (position + max_class_count) % len(samples)
        ]
    if any(labels[index] == labels[donor] for index, donor in enumerate(shuffle_indices)):
        raise RuntimeError("跨亚型错配构造失败")
    return samples, shuffle_indices


def run_audit(args):
    """对一个既有最佳检查点进行两次验证集推理。"""
    if args.batch_size <= 0:
        raise ValueError("批量大小必须为正数")
    device = torch.device(args.device)
    _validate_device(device)
    run_dir = Path(args.run_dir)
    saved_args = _load_json(run_dir / "args.json")
    manifest = _load_json(run_dir / "split_manifest.json")
    validate_flat5_checkpoint_args(saved_args, manifest)
    fusion_mode = saved_args.get("fusion_mode")
    if fusion_mode not in {"bus_text", "swe_residual"}:
        raise ValueError("仅支持本轮 BUS+Text 与 SWE 残差检查点")

    splits = restore_manifest_splits(
        build_fair_splits(
            args.project_root,
            "flat5",
            saved_args["malignant_metadata"],
            saved_args["benign_metadata"],
        ),
        manifest,
    )
    samples, shuffle_indices = prepare_validation_cases(splits)
    dataset = MultiModalBreastDataset(
        args.project_root,
        split="val",
        img_size=saved_args["img_size"],
        max_text_len=saved_args["max_text_len"],
        samples=samples,
        augment=False,
    )
    model = build_model(argparse.Namespace(**saved_args), device)
    checkpoint = torch.load(
        run_dir / "best_model.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    base_rows = _predict_dataset(model, dataset, device, args.batch_size)
    shuffled_rows = _predict_dataset(
        model,
        SWEPerturbationDataset(
            dataset, "case_shuffle", shuffle_indices=shuffle_indices
        ),
        device,
        args.batch_size,
    )
    if len(base_rows) != len(samples) or len(shuffled_rows) != len(samples):
        raise RuntimeError("验证集预测记录数与划分清单不一致")
    case_ids = [sample["case_id"] for sample in samples]
    rows = build_case_comparison_rows(
        base_rows,
        shuffled_rows,
        expected_donor_by_case={
            case_id: case_ids[shuffle_indices[index]]
            for index, case_id in enumerate(case_ids)
        },
        true_label_by_case={
            sample["case_id"]: int(sample["subtype_label"])
            for sample in samples
        },
    )
    metrics = summarize_condition_metrics(rows)
    changed_count = sum(row["prediction_changed"] for row in rows)
    summary = {
        "split": "malignant_val",
        "fusion_mode": fusion_mode,
        "run_dir": str(run_dir),
        "case_count": len(rows),
        "prediction_changed_count": changed_count,
        "prediction_changed_rate": changed_count / len(rows),
        "mean_abs_confidence_delta": sum(
            abs(row["confidence_delta"]) for row in rows
        ) / len(rows),
        "metrics": metrics,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="恶性验证集 SWE 病例错配审计")
    parser.add_argument("--project_root", required=True, help="包含 data 的项目根目录")
    parser.add_argument("--run_dir", required=True, help="已有最佳检查点所在目录")
    parser.add_argument("--output_dir", required=True, help="审计结果目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--batch_size", type=int, default=8, help="推理批量大小")
    return parser


if __name__ == "__main__":
    run_audit(build_parser().parse_args())
