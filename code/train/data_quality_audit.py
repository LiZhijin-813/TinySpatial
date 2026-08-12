"""执行乳腺癌分型数据的脱敏质量审计。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.datasets.split_utils import derive_suspected_group_id, load_metadata


TARGET_KEYWORDS = ("Luminal A", "Luminal B", "HER2", "TNBC")
MODALITIES = {
    "bus": "BUS",
    "swe": "SWE",
    "cdfi": "CDFI",
}


def classify_text_risk(raw_text: str) -> dict:
    """返回文本长度、空文本状态和目标关键词命中情况，不保留原文。"""
    text = "" if raw_text is None else str(raw_text)
    lowered = text.casefold()
    hits = [keyword for keyword in TARGET_KEYWORDS if keyword.casefold() in lowered]
    return {
        "text_length": len(text),
        "is_empty": not bool(text.strip()),
        "contains_target_keyword": bool(hits),
        "target_keyword_hits": hits,
    }


def audit_image_file(path: Path) -> dict:
    """返回图像可读性、尺寸、通道数和近似常量图像标记。"""
    path = Path(path)
    result = {
        "exists": path.is_file(),
        "readable": False,
        "width": None,
        "height": None,
        "channels": None,
        "pixel_std": None,
        "near_constant": False,
        "error": None,
    }
    if not result["exists"]:
        result["error"] = "文件不存在"
        return result
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
            result["readable"] = True
            result["width"], result["height"] = image.size
            result["channels"] = 3
            pixel_std = float(array.std())
            result["pixel_std"] = pixel_std
            result["near_constant"] = pixel_std < 1.0
    except (OSError, ValueError) as error:
        result["error"] = type(error).__name__
    return result


def build_quality_summary(
    samples: Sequence[Mapping],
    quality_rows: Sequence[Mapping],
) -> dict:
    """汇总标签、病例组、模态完整性和文本风险统计。"""
    label_counts = [0, 0, 0, 0]
    groups = defaultdict(list)
    for sample in samples:
        label = int(sample["subtype_label"])
        if 0 <= label < 4:
            label_counts[label] += 1
        group_id = str(sample.get("group_id") or derive_suspected_group_id(sample["case_id"]))
        groups[group_id].append(sample)

    group_conflicts = []
    for group_id, group_samples in sorted(groups.items()):
        labels = sorted({int(sample["subtype_label"]) for sample in group_samples})
        if len(labels) > 1:
            group_conflicts.append({"group_id": group_id, "labels": labels})

    modality_counts = {
        field: sum(bool(row.get(field)) for row in quality_rows)
        for field in ("bus_exists", "swe_exists", "cdfi_exists", "text_exists")
    }
    return {
        "sample_count": len(samples),
        "label_counts": label_counts,
        "group_count": len(groups),
        "multi_case_group_count": sum(len(values) > 1 for values in groups.values()),
        "label_conflict_group_count": len(group_conflicts),
        "label_conflict_groups": group_conflicts,
        "modality_available_counts": modality_counts,
        "target_keyword_case_count": sum(
            bool(row.get("contains_target_keyword")) for row in quality_rows
        ),
        "label_conflict_case_count": sum(
            bool(row.get("label_conflict")) for row in quality_rows
        ),
        "empty_text_case_count": sum(
            bool(row.get("text_is_empty")) for row in quality_rows
        ),
        "near_constant_image_count": sum(
            int(row.get("near_constant_image_count", 0)) for row in quality_rows
        ),
    }


def write_quality_outputs(
    output_dir: Path,
    summary: Mapping,
    rows: Sequence[Mapping],
) -> None:
    """写入脱敏 JSON、CSV 和中文 Markdown 报告。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_quality_summary.json").write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "case_quality_flags.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# 数据质量审计报告",
        "",
        f"- 病例数：{summary['sample_count']}",
        f"- 四类标签计数：{summary['label_counts']}",
        f"- 目标关键词病例数：{summary['target_keyword_case_count']}",
        f"- 标签冲突病例数：{summary['label_conflict_case_count']}",
        f"- 多病例疑似组数：{summary['multi_case_group_count']}",
        f"- 疑似标签冲突组数：{summary['label_conflict_group_count']}",
        "",
        "以上均为风险统计，不自动判定标签错误，也不包含原始文本。",
    ]
    (output_dir / "data_quality_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("raw_text", "")) if isinstance(data, dict) else ""


def audit_samples(project_root: Path, metadata_file: str) -> tuple[list[dict], dict]:
    """对恶性元数据执行一次全量脱敏审计。"""
    data_dir = Path(project_root) / "data"
    samples = load_metadata(data_dir / metadata_file, "malignant")
    groups = defaultdict(list)
    for sample in samples:
        groups[derive_suspected_group_id(sample["case_id"])].append(sample)
    rows = []
    for sample in samples:
        case_id = sample["case_id"]
        group_id = derive_suspected_group_id(case_id)
        group_labels = {item["subtype_label"] for item in groups[group_id]}
        text_risk = classify_text_risk(_read_text(data_dir / "texts" / f"{case_id}.json"))
        image_results = {
            modality: audit_image_file(
                data_dir / "images" / directory / f"{case_id}.jpg"
            )
            for modality, directory in MODALITIES.items()
        }
        rows.append(
            {
                "case_id": case_id,
                "split": sample["split"],
                "subtype_label": int(sample["subtype_label"]),
                "group_id": group_id,
                "label_conflict": len(group_labels) > 1,
                "bus_exists": image_results["bus"]["exists"],
                "swe_exists": image_results["swe"]["exists"],
                "cdfi_exists": image_results["cdfi"]["exists"],
                "text_exists": (data_dir / "texts" / f"{case_id}.json").is_file(),
                "text_is_empty": text_risk["is_empty"],
                "text_length": text_risk["text_length"],
                "contains_target_keyword": text_risk["contains_target_keyword"],
                "target_keyword_hits": ",".join(text_risk["target_keyword_hits"]),
                "near_constant_image_count": sum(
                    result["near_constant"] for result in image_results.values()
                ),
            }
        )
    return rows, build_quality_summary(samples, rows)


def build_parser() -> argparse.ArgumentParser:
    """构造数据质量审计命令行解析器。"""
    parser = argparse.ArgumentParser(description="乳腺癌分型数据质量审计")
    parser.add_argument("--project_root", default=str(PROJECT_ROOT), help="项目根目录")
    parser.add_argument("--metadata_file", default="metadata.csv", help="恶性元数据文件名")
    parser.add_argument("--case_audit_dir", help="已有病例审计目录，仅用于保留审计来源说明")
    parser.add_argument("--output_dir", required=True, help="脱敏审计输出目录")
    return parser


def main() -> None:
    """执行命令行数据质量审计。"""
    args = build_parser().parse_args()
    rows, summary = audit_samples(Path(args.project_root), args.metadata_file)
    summary = {**summary, "case_audit_dir": args.case_audit_dir}
    write_quality_outputs(Path(args.output_dir), summary, rows)
    print(f"数据质量审计完成，输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
