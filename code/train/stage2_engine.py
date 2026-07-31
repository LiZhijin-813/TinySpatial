"""Stage 2 单轮训练、任务感知评估和模型选择工具。"""

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


def _loader_length(loader, purpose):
    """读取并校验数据加载器批次数。"""
    try:
        batch_count = len(loader)
    except (TypeError, AttributeError) as error:
        raise ValueError(f"{purpose}数据加载器必须支持长度查询") from error
    if batch_count <= 0:
        raise ValueError(f"{purpose}数据加载器不能为空")
    return batch_count


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
    batch_count = _loader_length(loader, "训练")
    accum_steps = _validate_accum_steps(accum_steps)
    max_grad_norm = _validate_max_grad_norm(max_grad_norm)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {}
    sample_count = 0
    correct = 0
    processed_batches = 0

    for step, raw_batch in enumerate(
        tqdm(loader, desc="训练", leave=False)
    ):
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

        group_start = (step // accum_steps) * accum_steps
        group_size = min(accum_steps, batch_count - group_start)
        (losses["total_loss"] / group_size).backward()

        is_group_end = (step + 1) % accum_steps == 0
        is_last_batch = step + 1 == batch_count
        if is_group_end or is_last_batch:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(
                value.detach().item()
            )
        correct += int((predictions == labels).sum().item())
        sample_count += int(labels.numel())
        processed_batches += 1

    if processed_batches == 0:
        raise ValueError("训练数据加载器未产生任何 batch")

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
    _loader_length(loader, "评估")

    model.eval()
    labels = {"class": [], "subtype": [], "malignancy": []}
    logits = {"class": [], "subtype": [], "malignancy": []}
    image_features = []

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


def _select_metric_section(metrics, section_names):
    """按兼容优先级选择首个可用指标分区。"""
    for name in section_names:
        section = metrics.get(name)
        if section is not None:
            return section
    raise ValueError("指标中缺少可用的评估分区")


def monitor_value(metrics, monitor_metric):
    """按监控项读取模型选择数值。"""
    if monitor_metric == "malignant_macro_f1":
        return float(metrics["malignant"]["macro_f1"])
    if monitor_metric == "macro_f1":
        section = _select_metric_section(
            metrics,
            ("malignant", "overall", "binary"),
        )
        return float(section["macro_f1"])
    if monitor_metric == "balanced_acc":
        section = _select_metric_section(
            metrics,
            ("malignant", "overall", "binary"),
        )
        return float(section["balanced_accuracy"])
    if monitor_metric == "acc":
        section = _select_metric_section(
            metrics,
            ("overall", "malignant", "binary"),
        )
        return float(section["accuracy"])
    raise ValueError(f"未知 monitor_metric: {monitor_metric}")
