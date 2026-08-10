"""对多个固定种子的病例级特征探针执行无调参多数投票集成。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.train.feature_probe import fit_linear_probe
from code.utils.evaluation import SUBTYPE_NAMES, evaluate_predictions


def _load_feature_artifacts(feature_dir):
    """读取一个特征探针目录中的压缩特征文件。"""
    path = Path(feature_dir) / "fused_features.npz"
    if not path.is_file():
        raise ValueError(f"缺少特征文件：{path}")
    values = np.load(path)
    required = {
        "train_features",
        "train_labels",
        "val_features",
        "val_labels",
        "test_features",
        "test_labels",
    }
    if not required.issubset(values.files):
        raise ValueError(f"特征文件缺少必要字段：{path}")
    return {
        name: torch.from_numpy(values[name]).float()
        for name in required
    }


def _majority_vote(predictions, num_classes):
    """按样本执行多数投票，平票时选择编号较小的类别。"""
    stacked = torch.stack(predictions, dim=0)
    result = []
    for column in stacked.T:
        counts = torch.bincount(column, minlength=num_classes)
        result.append(int(counts.argmax()))
    return torch.tensor(result, dtype=torch.long)


def _metrics(labels, predictions):
    """将预测结果转换为统一的四分类指标。"""
    return evaluate_predictions(
        labels.numpy(),
        predictions.numpy(),
        SUBTYPE_NAMES,
    )


def run_ensemble(feature_dirs, output_dir):
    """执行多个种子的无调参线性探针多数投票。"""
    if len(feature_dirs) < 2:
        raise ValueError("集成至少需要两个特征目录")
    artifacts = [_load_feature_artifacts(path) for path in feature_dirs]
    reference = artifacts[0]
    for index, artifact in enumerate(artifacts[1:], start=1):
        for split in ("train", "val", "test"):
            if not torch.equal(artifact[f"{split}_labels"], reference[f"{split}_labels"]):
                raise ValueError(f"第 {index} 个特征目录的 {split} 标签顺序不一致")

    train_labels = reference["train_labels"].long()
    counts = torch.tensor(
        [int((train_labels == class_index).sum()) for class_index in range(4)],
        dtype=torch.float32,
    )
    inverse_weights = counts.sum() / (4.0 * counts)
    predictions = {"linear": {"val": [], "test": []}, "inverse": {"val": [], "test": []}}
    for artifact in artifacts:
        for split in ("val", "test"):
            predictions["linear"][split].append(
                fit_linear_probe(
                    artifact["train_features"],
                    train_labels,
                    artifact[f"{split}_features"],
                    num_classes=4,
                    epochs=300,
                    learning_rate=0.05,
                    weight_decay=1e-4,
                )
            )
            predictions["inverse"][split].append(
                fit_linear_probe(
                    artifact["train_features"],
                    train_labels,
                    artifact[f"{split}_features"],
                    num_classes=4,
                    epochs=300,
                    learning_rate=0.05,
                    weight_decay=1e-4,
                    class_weights=inverse_weights,
                )
            )

    metrics = {
        "protocol": "每个种子只使用训练特征拟合探针，验证集和测试集只做多数投票评估",
        "feature_directories": [str(path) for path in feature_dirs],
        "seed_count": len(feature_dirs),
        "ensemble": {"linear": {}, "inverse": {}},
    }
    for method in metrics["ensemble"]:
        for split in ("val", "test"):
            voted = _majority_vote(predictions[method][split], num_classes=4)
            metrics["ensemble"][method][split] = _metrics(
                reference[f"{split}_labels"].long(),
                voted,
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "feature_ensemble_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"特征探针集成完成，结果已保存到：{output_path}")
    return metrics


def build_parser():
    """构建特征探针集成命令行解析器。"""
    parser = argparse.ArgumentParser(description="病例级特征探针多数投票集成")
    parser.add_argument("--feature_dirs", nargs="+", required=True, help="特征探针目录")
    parser.add_argument("--output_dir", required=True, help="集成指标输出目录")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run_ensemble(arguments.feature_dirs, arguments.output_dir)
