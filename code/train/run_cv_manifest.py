"""生成病例组约束的五折分层交叉验证清单并输出数据审计。"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.datasets.split_utils import derive_suspected_group_id, load_metadata
from code.train.cv_splits import build_group_stratified_folds


def _group_audit(samples):
    """统计疑似病例组大小和组内标签冲突。"""
    groups = defaultdict(list)
    for sample in samples:
        groups[derive_suspected_group_id(sample["case_id"])].append(sample)
    conflicts = []
    for group_id, group_samples in sorted(groups.items()):
        labels = sorted({sample["subtype_label"] for sample in group_samples})
        if len(labels) > 1:
            conflicts.append(
                {
                    "group_id": group_id,
                    "case_ids": sorted(sample["case_id"] for sample in group_samples),
                    "labels": labels,
                }
            )
    return {
        "group_count": len(groups),
        "multi_case_group_count": sum(len(values) > 1 for values in groups.values()),
        "max_group_size": max(len(values) for values in groups.values()),
        "label_conflict_groups": conflicts,
    }


def _fold_summary(fold_index, fold):
    """生成单折病例数量、标签数量和组数量摘要。"""
    group_ids = {derive_suspected_group_id(sample["case_id"]) for sample in fold}
    return {
        "fold": fold_index,
        "case_count": len(fold),
        "class_counts": [
            sum(sample["subtype_label"] == label for sample in fold)
            for label in range(4)
        ],
        "group_count": len(group_ids),
        "case_ids": [sample["case_id"] for sample in fold],
        "group_ids": sorted(group_ids),
    }


def build_manifest(project_root, metadata_file, n_splits, seed):
    """读取恶性元数据并构造可复现的病例组分层清单。"""
    samples = load_metadata(Path(project_root) / "data" / metadata_file, "malignant")
    if any(sample["subtype_label"] < 0 for sample in samples):
        raise ValueError("恶性元数据包含无效亚型标签")
    folds = build_group_stratified_folds(samples, n_splits=n_splits, seed=seed)
    return {
        "protocol": "按疑似基础病例组约束的确定性亚型分层交叉验证",
        "metadata_file": metadata_file,
        "seed": seed,
        "n_splits": n_splits,
        "sample_count": len(samples),
        "overall_class_counts": [
            sum(sample["subtype_label"] == label for sample in samples)
            for label in range(4)
        ],
        "group_audit": _group_audit(samples),
        "folds": [_fold_summary(index, fold) for index, fold in enumerate(folds)],
    }


def build_parser():
    """构建病例组交叉验证清单命令行解析器。"""
    parser = argparse.ArgumentParser(description="生成病例组分层交叉验证清单")
    parser.add_argument("--project_root", default=str(PROJECT_ROOT), help="项目根目录")
    parser.add_argument("--metadata_file", default="metadata.csv", help="恶性元数据文件")
    parser.add_argument("--n_splits", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", required=True, help="JSON 清单输出路径")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    manifest = build_manifest(
        arguments.project_root,
        arguments.metadata_file,
        arguments.n_splits,
        arguments.seed,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"交叉验证清单已保存到：{output}")
