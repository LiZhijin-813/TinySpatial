"""固定检查点病例组五折层级探针主流程与纯函数。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.datasets.dataset import MultiModalBreastDataset
from code.datasets.split_utils import load_metadata
from code.train.feature_probe import fit_linear_probe
from code.train.run_cv_feature_probe import (
    _collect_features,
    _read_json,
    _resolve_device,
)
from code.train.train_stage2 import build_eval_loader, build_model
from code.utils.evaluation import evaluate_predictions


def build_hierarchical_tasks():
    """构建层级标签探针所需的三组二分类任务配置。"""
    return {
        "luminal_vs_non_luminal": {
            "positive_labels": (0, 1),
            "negative_labels": (2, 3),
            "display_names": ("Luminal", "非Luminal"),
        },
        "luminal_a_vs_luminal_b": {
            "positive_labels": (0,),
            "negative_labels": (1,),
            "display_names": ("Luminal A", "Luminal B"),
        },
        "her2_vs_tnbc": {
            "positive_labels": (2,),
            "negative_labels": (3,),
            "display_names": ("HER2+", "TNBC"),
        },
    }


def select_hierarchical_labels(labels, task):
    """筛选任务相关标签，并映射为二元标签。"""
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("标签必须是一维张量")
    if labels.dtype not in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("标签必须是整数张量")
    if torch.any((labels < 0) | (labels >= 4)):
        raise ValueError("标签必须处于 0 到 3 的四分类空间内")

    positive_labels = tuple(task["positive_labels"])
    negative_labels = tuple(task["negative_labels"])

    positive = torch.zeros_like(labels, dtype=torch.bool)
    negative = torch.zeros_like(labels, dtype=torch.bool)
    for label in positive_labels:
        positive |= labels == label
    for label in negative_labels:
        negative |= labels == label

    mask = positive | negative
    binary_labels = torch.where(positive[mask], 0, 1)
    return mask, binary_labels


def aggregate_binary_metrics(fold_metrics):
    """汇总二分类任务在各折上的指标均值、标准差与逐折数值。"""
    if not fold_metrics:
        raise ValueError("折级指标不能为空")

    aggregated = {}
    for metric_name in ("macro_f1", "balanced_accuracy"):
        values = [float(metrics[metric_name]) for metrics in fold_metrics]
        mean_value = sum(values) / len(values)
        variance = sum(
            (value - mean_value) ** 2 for value in values
        ) / len(values)
        aggregated[metric_name] = {
            "mean": float(mean_value),
            "std": float(math.sqrt(variance)),
            "values": [float(value) for value in values],
        }
    return aggregated


def validate_fold_labels(train_labels, query_labels, task_name):
    """校验训练折与测试折标签是否满足二分类探针要求。"""

    def _validate_tensor(labels, split_name):
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
            raise ValueError(f"{task_name}的{split_name}标签必须是一维整数张量")
        if labels.dtype not in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"{task_name}的{split_name}标签必须是一维整数张量")

        unique_labels = set(labels.tolist())
        if not {0, 1}.issubset(unique_labels):
            raise ValueError(f"{task_name}的{split_name}标签缺少二分类类别 0 或 1")

    _validate_tensor(train_labels, "训练折")
    _validate_tensor(query_labels, "测试折")


def _task_name(task):
    """构造用于日志与报错的中文任务名称。"""
    return f"{task['display_names'][0]} 对 {task['display_names'][1]}"


def _case_label_index(samples):
    """将恶性元数据转换为按病例编号索引的四分类标签。"""
    if not samples:
        raise ValueError("恶性元数据为空，无法执行层级探针")

    index = {}
    for sample in samples:
        case_id = sample.get("case_id")
        label = sample.get("subtype_label")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("恶性元数据包含无效病例编号")
        if case_id in index:
            raise ValueError(f"恶性元数据包含重复病例编号：{case_id}")
        if (
            isinstance(label, bool)
            or not isinstance(label, int)
            or label < 0
            or label >= 4
        ):
            raise ValueError("恶性元数据包含无效亚型标签")
        index[case_id] = label
    return index


def _normalize_manifest_folds(manifest, case_label_map):
    """校验病例组五折清单，并返回规范化后的折信息。"""
    if manifest.get("n_splits") != 5:
        raise ValueError("当前诊断要求五折病例组清单")

    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("病例组清单缺少五个验证折")

    normalized = []
    seen_case_ids = {}
    for default_fold, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise ValueError("病例组清单中的每一折都必须是对象")
        fold_id = fold.get("fold", default_fold)
        if isinstance(fold_id, bool) or not isinstance(fold_id, int):
            raise ValueError("病例组清单中的折编号必须是整数")
        fold_case_ids = fold.get("case_ids")
        if not isinstance(fold_case_ids, list) or not fold_case_ids:
            raise ValueError(f"第 {fold_id} 折缺少有效病例编号列表")
        if any(
            not isinstance(case_id, str) or not case_id.strip()
            for case_id in fold_case_ids
        ):
            raise ValueError(f"第 {fold_id} 折包含无效病例编号")
        if len(set(fold_case_ids)) != len(fold_case_ids):
            raise ValueError(f"第 {fold_id} 折存在重复病例编号")

        unknown_case_ids = [
            case_id for case_id in fold_case_ids if case_id not in case_label_map
        ]
        if unknown_case_ids:
            raise ValueError(
                f"第 {fold_id} 折包含未知病例编号：{unknown_case_ids[:5]}"
            )

        overlap_case_ids = [
            case_id for case_id in fold_case_ids if case_id in seen_case_ids
        ]
        if overlap_case_ids:
            raise ValueError(
                "病例组清单存在重复分配的病例编号："
                f"{overlap_case_ids[:5]}"
            )
        for case_id in fold_case_ids:
            seen_case_ids[case_id] = fold_id
        normalized.append({"fold": fold_id, "case_ids": list(fold_case_ids)})

    missing_case_ids = [
        case_id for case_id in case_label_map if case_id not in seen_case_ids
    ]
    if missing_case_ids:
        raise ValueError(
            f"病例组清单未覆盖全部恶性病例：{missing_case_ids[:5]}"
        )
    return normalized


def _validate_hierarchical_boundaries(folds, ordered_case_ids, case_label_map, tasks):
    """在加载权重前验证每个层级任务在每折上的标签边界。"""
    all_case_ids = set(ordered_case_ids)

    for task in tasks.values():
        task_name = _task_name(task)
        for fold in folds:
            fold_id = fold["fold"]
            query_case_ids = fold["case_ids"]
            query_case_id_set = set(query_case_ids)
            train_case_ids = [
                case_id for case_id in ordered_case_ids if case_id not in query_case_id_set
            ]
            if not train_case_ids:
                raise ValueError(f"第 {fold_id} 折缺少训练病例")
            if query_case_id_set - all_case_ids:
                raise ValueError(f"第 {fold_id} 折包含未收录于元数据的病例编号")

            train_labels = torch.tensor(
                [case_label_map[case_id] for case_id in train_case_ids],
                dtype=torch.long,
            )
            query_labels = torch.tensor(
                [case_label_map[case_id] for case_id in query_case_ids],
                dtype=torch.long,
            )
            train_mask, train_binary = select_hierarchical_labels(train_labels, task)
            query_mask, query_binary = select_hierarchical_labels(query_labels, task)
            if int(train_mask.sum()) == 0 or int(query_mask.sum()) == 0:
                raise ValueError(
                    f"{task_name}在第 {fold_id} 折的层级筛选结果为空"
                )
            validate_fold_labels(train_binary, query_binary, task_name)


def _build_binary_metrics(labels, predictions, task):
    """按层级任务类别名称计算统一二分类指标。"""
    return evaluate_predictions(
        labels.numpy(),
        predictions.numpy(),
        class_names=task["display_names"],
    )


def _binary_distribution(labels):
    """统计二分类标签分布。"""
    return torch.bincount(labels, minlength=2).tolist()


def _build_fold_record(fold_id, train_binary, query_binary, metrics):
    """构建逐折结果，保留显式的有效样本数与核心诊断字段。"""
    return {
        "fold": int(fold_id),
        "train_effective_count": int(train_binary.numel()),
        "query_effective_count": int(query_binary.numel()),
        "train_label_distribution": _binary_distribution(train_binary),
        "query_label_distribution": _binary_distribution(query_binary),
        "prediction_distribution": list(metrics["prediction_distribution"]),
        "confusion_matrix": metrics["confusion_matrix"],
        "macro_recall": float(metrics["macro_recall"]),
        "metrics": metrics,
    }


def run_cv_hierarchical_probe(
    run_dir,
    cv_manifest,
    output_dir,
    device_name="cuda:0",
    batch_size=16,
):
    """执行固定检查点病例组五折层级线性探针。"""
    run_path = Path(run_dir)
    saved_args = _read_json(run_path / "args.json")
    checkpoint_path = run_path / "best_model.pth"
    if saved_args.get("task_mode") != "flat5":
        raise ValueError("层级探针当前仅支持 flat5 检查点")
    if not checkpoint_path.is_file():
        raise ValueError(f"检查点不存在：{checkpoint_path}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")

    manifest = _read_json(cv_manifest)
    samples = load_metadata(
        PROJECT_ROOT / "data" / saved_args.get("malignant_metadata", "metadata.csv"),
        "malignant",
    )
    case_label_map = _case_label_index(samples)
    ordered_case_ids = [sample["case_id"] for sample in samples]
    folds = _normalize_manifest_folds(manifest, case_label_map)
    tasks = build_hierarchical_tasks()
    _validate_hierarchical_boundaries(
        folds,
        ordered_case_ids,
        case_label_map,
        tasks,
    )

    device = _resolve_device(device_name)
    dataset = MultiModalBreastDataset(
        PROJECT_ROOT,
        split="test",
        img_size=saved_args["img_size"],
        max_text_len=saved_args["max_text_len"],
        samples=samples,
        augment=False,
        ablate_modalities=saved_args.get("ablate_modalities", []),
    )
    model_args = argparse.Namespace(**saved_args)
    model = build_model(model_args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("检查点缺少 model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    loader_args = argparse.Namespace(**saved_args)
    loader_args.batch_size = batch_size
    loader_args.num_workers = saved_args.get("num_workers", 0)
    features, labels, case_ids = _collect_features(
        model,
        build_eval_loader(dataset, loader_args),
        device,
    )
    index = {case_id: position for position, case_id in enumerate(case_ids)}
    if len(index) != len(case_ids):
        raise ValueError("特征提取结果包含重复病例编号")

    missing_feature_case_ids = [
        case_id for case_id in ordered_case_ids if case_id not in index
    ]
    if missing_feature_case_ids:
        raise ValueError(
            "特征提取结果缺少病例编号："
            f"{missing_feature_case_ids[:5]}"
        )

    results = {
        "protocol": "固定检查点三层级病例组五折探针，仅使用训练折拟合二分类线性探针",
        "run_dir": str(run_path),
        "checkpoint": str(checkpoint_path),
        "cv_manifest": str(cv_manifest),
        "sample_count": len(case_ids),
        "feature_dimension": int(features.shape[1]),
        "tasks": {},
    }

    ordered_positions = list(range(len(case_ids)))
    for task_key, task in tasks.items():
        task_result = {
            "display_names": list(task["display_names"]),
            "methods": {
                "plain": [],
                "inverse_frequency": [],
            },
        }
        task_name = _task_name(task)
        for fold in folds:
            fold_id = fold["fold"]
            query_positions = [index[case_id] for case_id in fold["case_ids"]]
            query_position_set = set(query_positions)
            train_positions = [
                position
                for position in ordered_positions
                if position not in query_position_set
            ]

            train_features = features[train_positions]
            train_labels = labels[train_positions]
            query_features = features[query_positions]
            query_labels = labels[query_positions]

            train_mask, train_binary = select_hierarchical_labels(train_labels, task)
            query_mask, query_binary = select_hierarchical_labels(query_labels, task)
            if int(train_mask.sum()) == 0 or int(query_mask.sum()) == 0:
                raise ValueError(
                    f"{task_name}在第 {fold_id} 折的层级筛选结果为空"
                )
            validate_fold_labels(train_binary, query_binary, task_name)

            selected_train_features = train_features[train_mask]
            selected_query_features = query_features[query_mask]
            counts = torch.bincount(train_binary, minlength=2).float()
            weights = counts.sum() / (2.0 * counts)

            plain_predictions = fit_linear_probe(
                selected_train_features,
                train_binary,
                selected_query_features,
                num_classes=2,
                epochs=300,
                learning_rate=0.05,
                weight_decay=1e-4,
            )
            inverse_predictions = fit_linear_probe(
                selected_train_features,
                train_binary,
                selected_query_features,
                num_classes=2,
                epochs=300,
                learning_rate=0.05,
                weight_decay=1e-4,
                class_weights=weights,
            )

            plain_metrics = _build_binary_metrics(query_binary, plain_predictions, task)
            inverse_metrics = _build_binary_metrics(
                query_binary,
                inverse_predictions,
                task,
            )
            task_result["methods"]["plain"].append(
                _build_fold_record(
                    fold_id,
                    train_binary,
                    query_binary,
                    plain_metrics,
                )
            )
            task_result["methods"]["inverse_frequency"].append(
                _build_fold_record(
                    fold_id,
                    train_binary,
                    query_binary,
                    inverse_metrics,
                )
            )

        task_result["summary"] = {
            method_name: aggregate_binary_metrics(
                [record["metrics"] for record in fold_records]
            )
            for method_name, fold_records in task_result["methods"].items()
        }
        results["tasks"][task_key] = task_result

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "hierarchical_probe_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"层级探针完成，结果已保存到：{output_path}")
    return results


def build_parser():
    """构建病例组五折层级探针命令行解析器。"""
    parser = argparse.ArgumentParser(description="固定检查点病例组五折层级探针")
    parser.add_argument("--run_dir", required=True, help="已有训练运行目录")
    parser.add_argument("--cv_manifest", required=True, help="五折病例组清单")
    parser.add_argument("--output_dir", required=True, help="诊断输出目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--batch_size", type=int, default=16, help="推理批次大小")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run_cv_hierarchical_probe(
        arguments.run_dir,
        arguments.cv_manifest,
        arguments.output_dir,
        device_name=arguments.device,
        batch_size=arguments.batch_size,
    )
