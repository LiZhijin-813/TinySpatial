"""训练运行的轻量 JSON 产物管理。"""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


def _json_default(value):
    """将常见训练配置与数值对象转换为标准 JSON 类型。"""
    if isinstance(value, Namespace):
        return vars(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    raise TypeError(f"无法序列化类型: {type(value).__name__}")


def save_json(path, payload):
    """以 UTF-8 和可读缩进保存严格 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        content = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            default=_json_default,
        )
    except ValueError as error:
        raise ValueError(f"JSON 数据必须可序列化且不包含非有限数: {error}") from error
    path.write_text(content, encoding="utf-8")


def initialize_run_artifacts(output_dir, args, manifest):
    """初始化参数与数据划分清单产物。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        args_payload = vars(args)
    except TypeError as error:
        raise ValueError("args 必须是包含属性的参数命名空间") from error
    save_json(output_dir / "args.json", args_payload)
    save_json(output_dir / "split_manifest.json", manifest)


def save_training_state(output_dir, history, best_metrics):
    """保存训练历史与最佳模型指标。"""
    output_dir = Path(output_dir)
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "metrics_best.json", best_metrics)
