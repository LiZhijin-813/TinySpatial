"""从既有 Stage 2 检查点提取病例级融合特征并执行可分性诊断。"""

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

from code.datasets.dataset import MultiModalBreastDataset
from code.train.feature_probe import (
    fit_linear_probe,
    nearest_centroid_predict,
)
from code.train.train_stage2 import (
    build_eval_loader,
    build_fair_splits,
    build_model,
    restore_manifest_splits,
)
from code.utils.evaluation import SUBTYPE_NAMES, evaluate_predictions


def _read_json(path):
    """读取 UTF-8 JSON 对象。"""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 JSON 文件：{path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON 文件必须是对象：{path}")
    return value


def _resolve_device(name):
    """解析推理设备并检查 CUDA 可用性。"""
    try:
        device = torch.device(name)
    except (TypeError, RuntimeError) as error:
        raise ValueError("推理设备参数无效") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求了 CUDA 设备，但当前环境不可用")
    return device


def _build_dataset(samples, split, saved_args):
    """按保存的运行参数构造不启用随机增强的数据集。"""
    return MultiModalBreastDataset(
        PROJECT_ROOT,
        split=split,
        img_size=saved_args["img_size"],
        max_text_len=saved_args["max_text_len"],
        samples=samples,
        augment=False,
        ablate_modalities=saved_args.get("ablate_modalities", []),
    )


@torch.no_grad()
def _collect_features(model, loader, device):
    """按数据加载器顺序收集融合特征、标签和病例编号。"""
    model.eval()
    features = []
    labels = []
    case_ids = []
    for batch in loader:
        outputs = model(
            batch["bus_img"].to(device),
            batch["swe_img"].to(device),
            batch["cdfi_img"].to(device),
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        fused = outputs.get("fused_features")
        if not isinstance(fused, torch.Tensor) or fused.ndim != 2:
            raise ValueError("模型没有返回二维 fused_features")
        features.append(fused.detach().cpu())
        labels.append(batch["subtype_label"].detach().cpu())
        case_ids.extend(list(batch["case_id"]))
    if not features:
        raise ValueError("特征提取没有产生任何批次")
    return (
        torch.cat(features, dim=0),
        torch.cat(labels, dim=0).long(),
        case_ids,
    )


def _metric_summary(labels, predictions):
    """将探针预测转换为统一的四分类指标。"""
    return evaluate_predictions(
        labels.numpy(),
        predictions.numpy(),
        SUBTYPE_NAMES,
    )


def _geometry_summary(train_features, train_labels):
    """计算训练集类别中心及其两两距离，供判断少数类重叠程度。"""
    centroids = []
    for class_index in range(4):
        values = train_features[train_labels == class_index]
        if values.numel() == 0:
            raise ValueError(f"训练特征缺少亚型类别：{class_index}")
        centroids.append(values.mean(dim=0))
    centroid_tensor = torch.stack(centroids)
    distances = torch.cdist(centroid_tensor, centroid_tensor)
    return {
        "centroid_norms": [float(value) for value in centroid_tensor.norm(dim=1)],
        "centroid_distance_matrix": distances.tolist(),
        "tnbc_to_other_centroid_distances": [
            float(distances[3, index]) for index in range(3)
        ],
    }


def run_probe(run_dir, output_dir, device_name="cuda:0", batch_size=None):
    """执行一次固定检查点的训练集拟合、验证集测试和测试集评估。"""
    run_path = Path(run_dir)
    saved_args = _read_json(run_path / "args.json")
    manifest = _read_json(run_path / "split_manifest.json")
    if saved_args.get("task_mode") != "flat5" or manifest.get("task_mode") != "flat5":
        raise ValueError("特征探针当前仅支持 flat5 检查点")
    checkpoint_path = run_path / "best_model.pth"
    if not checkpoint_path.is_file():
        raise ValueError(f"检查点不存在：{checkpoint_path}")

    device = _resolve_device(device_name)
    canonical = build_fair_splits(
        PROJECT_ROOT,
        "flat5",
        malignant_metadata=saved_args.get("malignant_metadata", "metadata.csv"),
        benign_metadata=saved_args.get("benign_metadata", "metadata_5class.csv"),
    )
    restored = restore_manifest_splits(canonical, manifest)
    train_samples = [
        sample for sample in restored["train"] if sample["malignancy_label"] == 1
    ]
    split_samples = {
        "train": train_samples,
        "val": restored["malignant_val"],
        "test": restored["malignant_test"],
    }
    model_args = argparse.Namespace(**saved_args)
    model = build_model(model_args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("检查点缺少 model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    loader_args = argparse.Namespace(**saved_args)
    loader_args.batch_size = batch_size or saved_args["batch_size"]
    loader_args.num_workers = saved_args.get("num_workers", 0)
    extracted = {}
    for name, samples in split_samples.items():
        dataset = _build_dataset(samples, name, saved_args)
        extracted[name] = _collect_features(
            model,
            build_eval_loader(dataset, loader_args),
            device,
        )

    train_features, train_labels, train_case_ids = extracted["train"]
    metrics = {
        "run_dir": str(run_path),
        "checkpoint": str(checkpoint_path),
        "protocol": "仅使用恶性训练病例拟合探针，验证集和测试集只用于评估",
        "feature_name": "fused_features",
        "feature_dimension": int(train_features.shape[1]),
        "sample_counts": {
            name: int(values[0].shape[0]) for name, values in extracted.items()
        },
        "label_counts": {
            name: [
                int((values[1] == class_index).sum())
                for class_index in range(4)
            ]
            for name, values in extracted.items()
        },
        "geometry": _geometry_summary(train_features, train_labels),
        "nearest_centroid": {},
        "linear_probe": {},
        "linear_probe_inverse": {},
    }
    train_counts = torch.tensor(
        [int((train_labels == class_index).sum()) for class_index in range(4)],
        dtype=torch.float32,
    )
    inverse_weights = train_counts.sum() / (4.0 * train_counts)
    for name in ("val", "test"):
        query_features, query_labels, _ = extracted[name]
        centroid_predictions = nearest_centroid_predict(
            train_features,
            train_labels,
            query_features,
            num_classes=4,
        )
        linear_predictions = fit_linear_probe(
            train_features,
            train_labels,
            query_features,
            num_classes=4,
            epochs=300,
            learning_rate=0.05,
            weight_decay=1e-4,
        )
        inverse_predictions = fit_linear_probe(
            train_features,
            train_labels,
            query_features,
            num_classes=4,
            epochs=300,
            learning_rate=0.05,
            weight_decay=1e-4,
            class_weights=inverse_weights,
        )
        metrics["nearest_centroid"][name] = _metric_summary(
            query_labels,
            centroid_predictions,
        )
        metrics["linear_probe"][name] = _metric_summary(
            query_labels,
            linear_predictions,
        )
        metrics["linear_probe_inverse"][name] = _metric_summary(
            query_labels,
            inverse_predictions,
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path / "fused_features.npz",
        train_features=extracted["train"][0].numpy(),
        train_labels=extracted["train"][1].numpy(),
        val_features=extracted["val"][0].numpy(),
        val_labels=extracted["val"][1].numpy(),
        test_features=extracted["test"][0].numpy(),
        test_labels=extracted["test"][1].numpy(),
    )
    (output_path / "feature_probe_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "case_ids.json").write_text(
        json.dumps(
            {name: values[2] for name, values in extracted.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"特征探针完成，结果已保存到：{output_path}")
    return metrics


def build_parser():
    """构建特征探针命令行解析器。"""
    parser = argparse.ArgumentParser(description="病例级融合特征可分性诊断")
    parser.add_argument("--run_dir", required=True, help="已有训练运行目录")
    parser.add_argument("--output_dir", required=True, help="诊断产物目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--batch_size", type=int, help="推理批次大小")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run_probe(
        arguments.run_dir,
        arguments.output_dir,
        device_name=arguments.device,
        batch_size=arguments.batch_size,
    )
