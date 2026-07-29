"""实验数据划分、固定子集和疑似病例组审计工具。"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


VALID_TASK_MODES = {"overfit", "flat4", "flat5", "dual_head"}


def _normalize_row(row: Mapping[str, str], source: str) -> dict:
    class_label = int(row["subtype_label"])
    is_malignant = class_label < 4
    return {
        "case_id": row["case_id"],
        "class_label": class_label,
        "malignancy_label": int(is_malignant),
        "subtype_label": class_label if is_malignant else -1,
        "split": row["split"],
        "source": source,
    }


def load_metadata(path: str | Path, source: str) -> list[dict]:
    """读取元数据并转换为统一任务标签。"""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [_normalize_row(row, source) for row in csv.DictReader(handle)]


def _by_split(samples: Sequence[Mapping]) -> dict[str, list[dict]]:
    result = {"train": [], "val": [], "test": []}
    for sample in samples:
        result[sample["split"]].append(dict(sample))
    return result


def build_fair_splits(
    project_root: str | Path,
    task_mode: str,
    malignant_metadata: str = "metadata.csv",
    benign_metadata: str = "metadata_5class.csv",
) -> dict[str, list[dict]]:
    """按 canonical 恶性 A/B/C 划分构建可公平比较的任务划分。"""
    if task_mode not in VALID_TASK_MODES:
        raise ValueError(f"未知 task_mode: {task_mode}")

    data_dir = Path(project_root) / "data"
    malignant = _by_split(load_metadata(data_dir / malignant_metadata, "malignant"))
    five_class = load_metadata(data_dir / benign_metadata, "five_class")
    benign = _by_split([sample for sample in five_class if sample["class_label"] == 4])

    train = list(malignant["train"])
    if task_mode in {"flat5", "dual_head"}:
        train.extend(benign["train"])

    return {
        "train": train,
        "malignant_val": list(malignant["val"]),
        "malignant_test": list(malignant["test"]),
        "binary_val": list(malignant["val"]) + list(benign["val"]),
        "binary_test": list(malignant["test"]) + list(benign["test"]),
    }


def derive_suspected_group_id(case_id: str) -> str:
    """按首个连字符前缀推导疑似基础病例组，仅用于生成警告。"""
    return case_id.split("-", maxsplit=1)[0]


def audit_cross_split_groups(samples: Sequence[Mapping]) -> list[dict]:
    """列出跨 split 的疑似病例组，不修改任何样本归属。"""
    groups = defaultdict(list)
    for sample in samples:
        groups[derive_suspected_group_id(sample["case_id"])].append(sample)

    conflicts = []
    for group_id, group_samples in groups.items():
        splits = sorted({sample["split"] for sample in group_samples})
        if len(splits) > 1:
            conflicts.append({
                "suspected_group_id": group_id,
                "case_ids": sorted(sample["case_id"] for sample in group_samples),
                "splits": splits,
            })
    return sorted(conflicts, key=lambda item: item["suspected_group_id"])


def select_balanced_subset(
    samples: Sequence[Mapping],
    per_class: int,
    seed: int,
) -> list[dict]:
    """从恶性训练集的四个亚型中确定性抽取相同数量的样本。"""
    rng = random.Random(seed)
    selected = []
    for class_index in range(4):
        candidates = sorted(
            (dict(s) for s in samples if s["subtype_label"] == class_index),
            key=lambda sample: sample["case_id"],
        )
        if len(candidates) < per_class:
            raise ValueError(f"亚型 {class_index} 仅有 {len(candidates)} 例，少于 {per_class}")
        selected.extend(rng.sample(candidates, per_class))
    return sorted(selected, key=lambda sample: (sample["subtype_label"], sample["case_id"]))


def select_samples_by_case_ids(
    samples: Sequence[Mapping],
    case_ids: Sequence[str],
) -> list[dict]:
    """按清单顺序恢复样本，供检查点复现和固定子集恢复使用。"""
    index = {sample["case_id"]: dict(sample) for sample in samples}
    missing = [case_id for case_id in case_ids if case_id not in index]
    if missing:
        raise ValueError(f"划分清单包含未知 case_id: {missing[:5]}")
    return [index[case_id] for case_id in case_ids]


def build_split_manifest(task_mode: str, splits: Mapping, seed: int) -> dict:
    """生成不含图像或文本内容的轻量划分清单。"""
    all_samples = [sample for values in splits.values() for sample in values]
    return {
        "task_mode": task_mode,
        "seed": seed,
        "splits": {
            name: [sample["case_id"] for sample in values]
            for name, values in splits.items()
        },
        "suspected_cross_split_groups": audit_cross_split_groups(all_samples),
    }
