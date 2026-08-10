"""在病例组五折清单上评估固定检查点融合特征的线性可分性。"""

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
from code.datasets.split_utils import load_metadata
from code.train.feature_probe import fit_linear_probe
from code.train.train_stage2 import build_eval_loader, build_model
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
    """解析推理设备。"""
    try:
        device = torch.device(name)
    except (TypeError, RuntimeError) as error:
        raise ValueError("推理设备参数无效") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求的 CUDA 设备不可用")
    return device


@torch.no_grad()
def _collect_features(model, loader, device):
    """收集所有恶性病例的融合特征并保持数据集顺序。"""
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
        labels.append(batch["subtype_label"].detach().cpu().long())
        case_ids.extend(list(batch["case_id"]))
    if not features:
        raise ValueError("特征提取没有产生任何批次")
    return torch.cat(features), torch.cat(labels), case_ids


def _metrics(labels, predictions):
    """转换为统一四分类指标。"""
    return evaluate_predictions(
        labels.numpy(),
        predictions.numpy(),
        SUBTYPE_NAMES,
    )


def _aggregate(fold_metrics):
    """计算各折 Macro-F1 和平衡准确率的均值与标准差。"""
    summary = {}
    for name in ("macro_f1", "balanced_accuracy"):
        values = np.asarray([item[name] for item in fold_metrics], dtype=float)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "values": [float(value) for value in values],
        }
    return summary


def run_cv_probe(run_dir, cv_manifest, output_dir, device_name="cuda:0", batch_size=16):
    """提取固定检查点特征，并在病例组五折上拟合线性探针。"""
    run_path = Path(run_dir)
    saved_args = _read_json(run_path / "args.json")
    checkpoint_path = run_path / "best_model.pth"
    manifest = _read_json(cv_manifest)
    if saved_args.get("task_mode") != "flat5":
        raise ValueError("交叉验证特征探针当前仅支持 flat5 检查点")
    if manifest.get("n_splits") != 5:
        raise ValueError("当前诊断要求五折病例组清单")

    device = _resolve_device(device_name)
    samples = load_metadata(
        PROJECT_ROOT / "data" / saved_args.get("malignant_metadata", "metadata.csv"),
        "malignant",
    )
    if any(sample["subtype_label"] < 0 for sample in samples):
        raise ValueError("恶性元数据包含无效亚型标签")
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
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("病例组清单缺少五个验证折")

    results = {
        "protocol": "固定检查点融合特征，按病例组五折仅用训练折拟合线性探针",
        "run_dir": str(run_path),
        "cv_manifest": str(cv_manifest),
        "sample_count": len(case_ids),
        "methods": {"linear": [], "inverse": []},
    }
    all_positions = set(range(len(case_ids)))
    for fold in folds:
        fold_case_ids = fold.get("case_ids")
        if not isinstance(fold_case_ids, list) or not fold_case_ids:
            raise ValueError("交叉验证折缺少病例编号")
        test_positions = [index[case_id] for case_id in fold_case_ids]
        train_positions = sorted(all_positions - set(test_positions))
        train_features = features[train_positions]
        train_labels = labels[train_positions]
        query_features = features[test_positions]
        query_labels = labels[test_positions]
        counts = torch.tensor(
            [int((train_labels == class_index).sum()) for class_index in range(4)],
            dtype=torch.float32,
        )
        inverse_weights = counts.sum() / (4.0 * counts)
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
        results["methods"]["linear"].append(
            {"fold": fold["fold"], **_metrics(query_labels, linear_predictions)}
        )
        results["methods"]["inverse"].append(
            {"fold": fold["fold"], **_metrics(query_labels, inverse_predictions)}
        )

    results["summary"] = {
        method: _aggregate(fold_metrics)
        for method, fold_metrics in results["methods"].items()
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "cv_feature_probe_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"交叉验证特征探针完成，结果已保存到：{output_path}")
    return results


def build_parser():
    """构建交叉验证特征探针命令行解析器。"""
    parser = argparse.ArgumentParser(description="病例组五折特征探针诊断")
    parser.add_argument("--run_dir", required=True, help="已有训练运行目录")
    parser.add_argument("--cv_manifest", required=True, help="五折病例组清单")
    parser.add_argument("--output_dir", required=True, help="诊断输出目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--batch_size", type=int, default=16, help="推理批次大小")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run_cv_probe(
        arguments.run_dir,
        arguments.cv_manifest,
        arguments.output_dir,
        device_name=arguments.device,
        batch_size=arguments.batch_size,
    )
