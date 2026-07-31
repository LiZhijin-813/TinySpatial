"""Stage 2 单轮训练、任务感知评估和模型选择工具。"""

from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real

import torch
from tqdm import tqdm

from code.train.stage2_objectives import compute_stage2_losses
from code.utils.evaluation import (
    conditional_malignant_predictions,
    end_to_end_dual_predictions,
    end_to_end_flat5_predictions,
    evaluate_predictions,
)


SUBTYPE_NAMES = ["Luminal A", "Luminal B", "HER2+", "TNBC"]
FIVE_CLASS_NAMES = SUBTYPE_NAMES + ["Benign"]
BINARY_NAMES = ["Benign", "Malignant"]

_TASK_SPLITS = {
    "overfit": {"malignant"},
    "flat4": {"malignant"},
    "flat5": {"malignant", "binary"},
    "dual_head": {"malignant", "binary"},
}


def _validate_task_mode(task_mode):
    """校验 Stage 2 任务模式。"""
    if task_mode not in _TASK_SPLITS:
        raise ValueError(f"未知 task_mode: {task_mode}")


def _validate_accum_steps(accum_steps):
    """校验梯度累积步数为非布尔正整数。"""
    if (
        isinstance(accum_steps, bool)
        or not isinstance(accum_steps, Integral)
        or accum_steps <= 0
    ):
        raise ValueError("accum_steps 必须为非布尔正整数")
    return int(accum_steps)


def _validate_max_grad_norm(max_grad_norm):
    """校验梯度裁剪阈值为非负有限数。"""
    if (
        isinstance(max_grad_norm, bool)
        or not isinstance(max_grad_norm, Real)
        or not isfinite(max_grad_norm)
        or max_grad_norm < 0
    ):
        raise ValueError("max_grad_norm 必须为非负有限数")
    return float(max_grad_norm)


def move_batch_to_device(batch, device):
    """仅将批次中的张量迁移到目标设备。"""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _step_accumulated_gradients(
    model,
    optimizer,
    group_count,
    max_grad_norm,
):
    """按真实累积批次数归一化梯度并执行一次优化。"""
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(group_count)
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criteria,
    task_mode,
    device,
    lambda_bm=0.3,
    max_grad_norm=1.0,
    accum_steps=1,
):
    """按任务模式执行一个训练轮次并返回未缩放损失均值。"""
    _validate_task_mode(task_mode)
    accum_steps = _validate_accum_steps(accum_steps)
    max_grad_norm = _validate_max_grad_norm(max_grad_norm)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {}
    sample_count = 0
    correct = 0
    processed_batches = 0
    group_count = 0

    for raw_batch in tqdm(loader, desc="训练", leave=False):
        batch = move_batch_to_device(raw_batch, device)
        if task_mode == "overfit":
            outputs = model(batch["bus_img"])
            losses = {
                "total_loss": criteria["class"](
                    outputs["class_logits"],
                    batch["subtype_label"],
                )
            }
            predictions = outputs["class_logits"].argmax(dim=1)
            labels = batch["subtype_label"]
        else:
            outputs = model(
                batch["bus_img"],
                batch["swe_img"],
                batch["cdfi_img"],
                batch["input_ids"],
                batch["attention_mask"],
            )
            losses = compute_stage2_losses(
                outputs,
                batch,
                task_mode,
                criteria,
                lambda_bm=lambda_bm,
            )
            if task_mode == "flat4":
                predictions = outputs["class_logits"].argmax(dim=1)
                labels = batch["subtype_label"]
            elif task_mode == "flat5":
                predictions = outputs["class_logits"].argmax(dim=1)
                labels = batch["class_label"]
            else:
                malignant = batch["malignancy_label"] == 1
                predictions = outputs["subtype_logits"][malignant].argmax(dim=1)
                labels = batch["subtype_label"][malignant]

        losses["total_loss"].backward()
        group_count += 1

        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(
                value.detach().item()
            )
        correct += int((predictions == labels).sum().item())
        sample_count += int(labels.numel())
        processed_batches += 1

        if group_count == accum_steps:
            _step_accumulated_gradients(
                model,
                optimizer,
                group_count,
                max_grad_norm,
            )
            group_count = 0

    if processed_batches == 0:
        raise ValueError("训练数据加载器未产生任何 batch")
    if group_count > 0:
        _step_accumulated_gradients(
            model,
            optimizer,
            group_count,
            max_grad_norm,
        )

    return {
        **{
            name: value / processed_batches
            for name, value in totals.items()
        },
        "accuracy": correct / max(sample_count, 1),
    }


def _cat(values, name):
    """拼接评估张量并对缺失数据给出中文错误。"""
    if not values:
        raise ValueError(f"评估结果缺少{name}")
    return torch.cat(values, dim=0)


def _population_variance(values, name):
    """计算批次维度上的有限 population 方差均值。"""
    merged = _cat(values, name)
    variance = float(
        merged.var(dim=0, unbiased=False).mean().item()
    )
    if not isfinite(variance):
        raise ValueError(f"{name}方差必须为有限数")
    return variance


def _evaluation_diagnostic(logits, image_features):
    """汇总不参与模型选择的 logits 与特征方差诊断。"""
    diagnostic = {}
    merged_logits = [
        _cat(values, f"{name}_logits")
        for name, values in logits.items()
        if values
    ]
    if merged_logits:
        diagnostic["logits_var"] = _population_variance(
            [torch.cat(merged_logits, dim=1)],
            "logits",
        )
    if image_features:
        diagnostic["feature_var"] = _population_variance(
            image_features,
            "image_features",
        )
    return diagnostic


@torch.no_grad()
def evaluate_loader(model, loader, task_mode, device, split_kind):
    """按任务与数据集语义执行评估并返回结构化指标。"""
    _validate_task_mode(task_mode)
    if split_kind not in _TASK_SPLITS[task_mode]:
        raise ValueError(
            f"task_mode={task_mode} 不支持 split_kind={split_kind}"
        )

    model.eval()
    labels = {"class": [], "subtype": [], "malignancy": []}
    logits = {"class": [], "subtype": [], "malignancy": []}
    image_features = []
    processed_batches = 0

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        if task_mode == "overfit":
            outputs = model(batch["bus_img"])
        else:
            outputs = model(
                batch["bus_img"],
                batch["swe_img"],
                batch["cdfi_img"],
                batch["input_ids"],
                batch["attention_mask"],
            )

        for name in labels:
            label_key = f"{name}_label"
            logit_key = f"{name}_logits"
            if label_key in batch:
                labels[name].append(batch[label_key].detach().cpu())
            if logit_key in outputs:
                logits[name].append(outputs[logit_key].detach().cpu())
        if "image_features" in outputs:
            image_features.append(
                outputs["image_features"].detach().cpu()
            )
        processed_batches += 1

    if processed_batches == 0:
        raise ValueError("评估数据加载器未产生任何 batch")

    diagnostic = _evaluation_diagnostic(logits, image_features)

    if task_mode in {"overfit", "flat4"}:
        class_logits = _cat(logits["class"], "class_logits")
        y_true = _cat(labels["subtype"], "subtype_label").numpy()
        malignant = evaluate_predictions(
            y_true,
            class_logits.argmax(dim=1).numpy(),
            SUBTYPE_NAMES,
        )
        return {
            "malignant": malignant,
            "diagnostic": diagnostic,
        }

    if task_mode == "flat5":
        class_logits = _cat(logits["class"], "class_logits")
        if split_kind == "malignant":
            y_true = _cat(labels["subtype"], "subtype_label").numpy()
            conditional = evaluate_predictions(
                y_true,
                conditional_malignant_predictions(class_logits),
                SUBTYPE_NAMES,
            )
            end_to_end = evaluate_predictions(
                y_true,
                end_to_end_flat5_predictions(class_logits),
                FIVE_CLASS_NAMES,
                focus_labels=[0, 1, 2, 3],
            )
            return {
                "malignant": conditional,
                "malignant_end_to_end": end_to_end,
                "diagnostic": diagnostic,
            }

        y_true = _cat(
            labels["malignancy"],
            "malignancy_label",
        ).numpy()
        probabilities = class_logits.softmax(dim=1)
        binary_predictions = (
            class_logits.argmax(dim=1) != 4
        ).long().numpy()
        binary = evaluate_predictions(
            y_true,
            binary_predictions,
            BINARY_NAMES,
            y_score=probabilities[:, :4].sum(dim=1).numpy(),
        )
        return {"binary": binary, "diagnostic": diagnostic}

    malignancy_logits = _cat(
        logits["malignancy"],
        "malignancy_logits",
    )
    subtype_logits = _cat(logits["subtype"], "subtype_logits")
    if split_kind == "malignant":
        y_true = _cat(labels["subtype"], "subtype_label").numpy()
        conditional = evaluate_predictions(
            y_true,
            subtype_logits.argmax(dim=1).numpy(),
            SUBTYPE_NAMES,
        )
        end_to_end = evaluate_predictions(
            y_true,
            end_to_end_dual_predictions(
                malignancy_logits,
                subtype_logits,
            ),
            FIVE_CLASS_NAMES,
            focus_labels=[0, 1, 2, 3],
        )
        return {
            "malignant": conditional,
            "malignant_end_to_end": end_to_end,
            "diagnostic": diagnostic,
        }

    y_true = _cat(
        labels["malignancy"],
        "malignancy_label",
    ).numpy()
    probabilities = malignancy_logits.softmax(dim=1)[:, 1].numpy()
    binary = evaluate_predictions(
        y_true,
        malignancy_logits.argmax(dim=1).numpy(),
        BINARY_NAMES,
        y_score=probabilities,
    )
    return {"binary": binary, "diagnostic": diagnostic}


def _metric_value(metrics, section_names, metric_key):
    """按分区优先级读取首个非空映射中的有限实数指标。"""
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics 必须是指标映射")

    for section_name in section_names:
        if section_name not in metrics:
            continue
        section = metrics[section_name]
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise ValueError(f"指标分区 {section_name} 必须是映射")
        if not section:
            continue
        if metric_key not in section:
            raise ValueError(
                f"指标分区 {section_name} 缺少 {metric_key}"
            )

        value = section[metric_key]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
        ):
            raise ValueError(
                f"指标 {section_name}.{metric_key} 必须是有限实数"
            )
        return float(value)

    raise ValueError(
        f"metrics 缺少包含 {metric_key} 的可用指标分区"
    )


def monitor_value(
    metrics,
    monitor_metric,
    allow_diagnostic=False,
):
    """安全读取主模型选择指标或显式授权的诊断指标。"""
    if not isinstance(allow_diagnostic, bool):
        raise ValueError("allow_diagnostic 必须是布尔值")

    metric_specs = {
        "malignant_macro_f1": (
            ("malignant",),
            "macro_f1",
        ),
        "macro_f1": (
            ("malignant", "overall", "binary"),
            "macro_f1",
        ),
        "balanced_acc": (
            ("malignant", "overall", "binary"),
            "balanced_accuracy",
        ),
        "acc": (
            ("overall", "malignant", "binary"),
            "accuracy",
        ),
    }
    if monitor_metric not in metric_specs:
        raise ValueError(f"未知 monitor_metric: {monitor_metric}")
    if monitor_metric != "malignant_macro_f1" and not allow_diagnostic:
        raise ValueError(
            "默认模型选择仅允许 malignant_macro_f1，"
            "诊断指标必须显式启用"
        )

    section_names, metric_key = metric_specs[monitor_metric]
    return _metric_value(metrics, section_names, metric_key)
