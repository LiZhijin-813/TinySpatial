"""Stage 2 可靠基线、过拟合门禁与检查点复评统一入口。"""

import argparse
import datetime
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.datasets.bus_dataset import BUSOverfitDataset
from code.datasets.dataset import (
    MultiModalBreastDataset,
    VALID_ABLATION_MODALITIES,
)
from code.datasets.split_utils import (
    build_fair_splits,
    build_split_manifest,
    select_balanced_subset,
    select_samples_by_case_ids,
)
from code.models.stage2.overfit_probe import BUSOverfitProbe
from code.models.stage2.subtyping_model import SECSubtypingModel
from code.train.run_artifacts import (
    initialize_run_artifacts,
    save_json,
    save_training_state,
)
from code.train.stage2_engine import (
    evaluate_loader,
    monitor_value,
    train_one_epoch,
)
from code.train.stage2_objectives import compute_class_weights
from code.utils.seed import seed_everything


TASK_MODES = ("overfit", "flat4", "flat5", "dual_head")
MONITOR_METRICS = (
    "malignant_macro_f1",
    "macro_f1",
    "balanced_acc",
    "acc",
)


def _translate_argparse_error(message):
    """翻译 argparse 常见错误，同时保留参数和值线索。"""
    invalid_choice = re.fullmatch(
        r"argument (?P<argument>\S+): invalid choice: "
        r"(?P<value>.+?) \(choose from (?P<choices>.+)\)",
        message,
    )
    if invalid_choice:
        return (
            f"参数 {invalid_choice['argument']} 的值 "
            f"{invalid_choice['value']} 无效，"
            f"允许值：{invalid_choice['choices']}"
        )

    required_prefix = "the following arguments are required: "
    if message.startswith(required_prefix):
        return f"缺少必需参数：{message[len(required_prefix):]}"

    unrecognized_prefix = "unrecognized arguments: "
    if message.startswith(unrecognized_prefix):
        return f"无法识别参数：{message[len(unrecognized_prefix):]}"

    invalid_value = re.fullmatch(
        r"argument (?P<argument>\S+): invalid (?P<kind>\S+) value: "
        r"(?P<value>.+)",
        message,
    )
    if invalid_value:
        kind_names = {
            "int": "整数",
            "float": "浮点数",
        }
        kind = kind_names.get(invalid_value["kind"], "指定类型")
        return (
            f"参数 {invalid_value['argument']} 的值 "
            f"{invalid_value['value']} 不是有效{kind}"
        )

    return "请检查参数名称、取值和格式"


class ChineseArgumentParser(argparse.ArgumentParser):
    """将命令行解析失败统一转换为中文提示。"""

    def error(self, message):
        """以中文参数错误结束解析。"""
        self.exit(2, f"参数错误：{_translate_argparse_error(message)}\n")


def build_parser():
    """构建可靠 Stage 2 训练与复评命令行解析器。"""
    parser = ChineseArgumentParser(description="Stage 2 可靠基线训练")
    parser.add_argument("--pretrained_path", required=True, help="预训练权重路径")
    parser.add_argument(
        "--task_mode",
        choices=TASK_MODES,
        default="flat4",
        help="训练任务模式",
    )
    parser.add_argument(
        "--flat5_class_weighting",
        choices=["inverse", "none"],
        default="inverse",
        help="flat5 五分类交叉熵类别权重策略",
    )
    parser.add_argument(
        "--malignant_metadata",
        default="metadata.csv",
        help="恶性病例元数据文件名",
    )
    parser.add_argument(
        "--benign_metadata",
        default="metadata_5class.csv",
        help="含良性病例的元数据文件名",
    )
    parser.add_argument(
        "--overfit_samples",
        type=int,
        choices=[32, 64],
        default=32,
        help="过拟合门禁样本数",
    )
    parser.add_argument(
        "--lambda_bm",
        type=float,
        default=0.3,
        help="双头良恶性损失固定权重",
    )
    parser.add_argument(
        "--sampler",
        choices=["none", "balanced"],
        default="none",
        help="训练采样策略",
    )
    parser.add_argument(
        "--monitor_metric",
        choices=MONITOR_METRICS,
        default="malignant_macro_f1",
        help="模型选择监控指标",
    )
    parser.add_argument(
        "--eval_malignant_subset",
        dest="eval_malignant_subset",
        action="store_true",
        help="评估固定恶性子集",
    )
    parser.add_argument(
        "--no_eval_malignant_subset",
        "--no-eval-malignant-subset",
        dest="eval_malignant_subset",
        action="store_false",
        help="不评估固定恶性子集",
    )
    parser.add_argument("--evaluate_checkpoint", help="仅复评指定检查点")
    parser.add_argument(
        "--eval_output",
        default="metrics_eval.json",
        help="检查点复评指标输出路径",
    )
    parser.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
        help="启用训练增强",
    )
    parser.add_argument(
        "--no_augment",
        "--no-augment",
        dest="augment",
        action="store_false",
        help="禁用训练增强",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="标签平滑系数，可靠基线固定为零",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="对齐损失权重，可靠基线固定为零",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="特征正则权重，可靠基线固定为零",
    )
    parser.add_argument("--img_size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-4, help="基础学习率")
    parser.add_argument("--wd", type=float, default=0.05, help="权重衰减")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="提前停止耐心轮数",
    )
    parser.add_argument(
        "--max_text_len",
        type=int,
        default=128,
        help="文本最大长度",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据加载进程数",
    )
    parser.add_argument(
        "--unfreeze",
        type=int,
        default=2,
        help="解冻编码器末尾层数",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="cuda:0", help="计算设备")
    parser.add_argument(
        "--output_root",
        default="runs/stage2",
        help="运行产物根目录",
    )
    parser.add_argument(
        "--ablate_modalities",
        nargs="*",
        choices=VALID_ABLATION_MODALITIES,
        default=[],
        help="实验时屏蔽的模态列表",
    )
    parser.set_defaults(augment=None, eval_malignant_subset=True)
    return parser


def validate_reliable_configuration(args):
    """拒绝改变可靠基线目标函数和模型选择口径的参数。"""
    for name in ("beta", "gamma", "label_smoothing"):
        if getattr(args, name) != 0.0:
            raise ValueError(f"可靠基线要求 {name}=0，不允许启用额外损失")
    if args.lambda_bm != 0.3:
        raise ValueError("dual_head 的 lambda_bm 固定为 0.3")
    if args.monitor_metric != "malignant_macro_f1":
        raise ValueError(
            "monitor_metric 仅允许 malignant_macro_f1，"
            "其他指标只能作为诊断输出"
        )
    if args.task_mode != "overfit" and not args.eval_malignant_subset:
        raise ValueError("可靠基线必须评估固定恶性子集")
    if args.task_mode != "flat5" and args.flat5_class_weighting == "none":
        raise ValueError(
            "flat5_class_weighting=none 仅允许 task_mode=flat5"
        )


def build_criteria_for_mode(
    task_mode,
    samples,
    device,
    label_smoothing=0.0,
    flat5_class_weighting="inverse",
):
    """按任务标签空间构造彼此独立的交叉熵损失。"""
    if task_mode == "overfit":
        return {"class": nn.CrossEntropyLoss()}
    if task_mode == "flat4":
        weights = compute_class_weights(
            samples,
            "subtype_label",
            4,
            device=device,
        )
        return {
            "class": nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=label_smoothing,
            )
        }
    if task_mode == "flat5":
        if flat5_class_weighting == "inverse":
            weights = compute_class_weights(
                samples,
                "class_label",
                5,
                device=device,
            )
        elif flat5_class_weighting == "none":
            weights = None
        else:
            raise ValueError(
                f"未知 flat5_class_weighting 策略：{flat5_class_weighting}"
            )
        return {
            "class": nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=label_smoothing,
            )
        }
    if task_mode != "dual_head":
        raise ValueError(f"未知 task_mode: {task_mode}")
    return {
        "malignancy": nn.CrossEntropyLoss(
            weight=compute_class_weights(
                samples,
                "malignancy_label",
                2,
                device=device,
            ),
            label_smoothing=label_smoothing,
        ),
        "subtype": nn.CrossEntropyLoss(
            weight=compute_class_weights(
                samples,
                "subtype_label",
                4,
                ignore_index=-1,
                device=device,
            ),
            label_smoothing=label_smoothing,
        ),
    }


def build_train_loader(dataset, task_mode, args):
    """构建随机打乱或按当前任务标签逆频率采样的训练加载器。"""
    sampler = None
    if args.sampler == "balanced":
        if task_mode == "overfit":
            raise ValueError(
                "过拟合门禁已经分层均衡，不允许重复使用 balanced sampler"
            )
        label_key = "subtype_label" if task_mode == "flat4" else "class_label"
        counts = Counter(int(sample[label_key]) for sample in dataset.samples)
        sample_weights = [
            1.0 / counts[int(sample[label_key])]
            for sample in dataset.samples
        ]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def build_eval_loader(dataset, args):
    """构建训练和检查点复评共用的确定性数据加载器。"""
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def _append_parameter_group(
    groups,
    seen,
    module,
    name,
    learning_rate,
    weight_decay,
):
    """追加模块中尚未收集的可训练参数。"""
    parameters = []
    for parameter in module.parameters():
        parameter_id = id(parameter)
        if parameter.requires_grad and parameter_id not in seen:
            seen.add(parameter_id)
            parameters.append(parameter)
    if parameters:
        groups.append({
            "params": parameters,
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "name": name,
        })


def build_optimizer(model, base_lr, weight_decay):
    """按模块分组构建 AdamW，并验证可训练参数无遗漏、无重复。"""
    groups = []
    seen = set()
    _append_parameter_group(
        groups,
        seen,
        model.task_heads,
        "task_heads",
        base_lr * 3,
        0.01,
    )
    _append_parameter_group(
        groups,
        seen,
        model.ip_adapters,
        "ip_adapters",
        base_lr * 2,
        weight_decay,
    )
    _append_parameter_group(
        groups,
        seen,
        model.cross_attention,
        "cross_attention",
        base_lr * 2,
        weight_decay,
    )
    _append_parameter_group(
        groups,
        seen,
        model.cdfi_branch,
        "cdfi_branch",
        base_lr * 2,
        weight_decay,
    )
    for index, block in enumerate(model.encoder.blocks):
        _append_parameter_group(
            groups,
            seen,
            block,
            f"encoder_block_{index}",
            base_lr,
            weight_decay,
        )

    encoder_shell = nn.ModuleList([
        model.encoder.patch_embed,
        model.encoder.norm,
    ])
    _append_parameter_group(
        groups,
        seen,
        encoder_shell,
        "patch_embed_norm",
        base_lr,
        weight_decay,
    )
    _append_parameter_group(
        groups,
        seen,
        model.text_branch,
        "text_projection",
        base_lr * 2,
        weight_decay,
    )

    optimizer = torch.optim.AdamW(groups)
    optimized_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if len(optimized_ids) != len(set(optimized_ids)):
        raise RuntimeError("优化器参数组包含重复参数")
    if set(optimized_ids) != expected:
        raise RuntimeError("优化器参数组与可训练参数不一致")
    return optimizer


def build_model(args, device):
    """按任务模式构造训练和复评共用模型。"""
    if args.task_mode == "overfit":
        model = BUSOverfitProbe(
            pretrained_path=args.pretrained_path,
            img_size=args.img_size,
        )
    else:
        model = SECSubtypingModel(
            pretrained_path=args.pretrained_path,
            task_mode=args.task_mode,
            img_size=args.img_size,
            unfreeze_last_n=args.unfreeze,
        )
    return model.to(device)


def restore_manifest_splits(canonical_splits, manifest):
    """从 canonical 样本池按 manifest 名称和 case ID 顺序恢复划分。"""
    candidates = []
    seen = set()
    for samples in canonical_splits.values():
        for sample in samples:
            if sample["case_id"] not in seen:
                seen.add(sample["case_id"])
                candidates.append(sample)
    return {
        name: select_samples_by_case_ids(candidates, case_ids)
        for name, case_ids in manifest["splits"].items()
    }


def load_checkpoint_run_configuration(args):
    """读取检查点相邻参数与清单，并校验复评任务模式。"""
    checkpoint_path = Path(args.evaluate_checkpoint)
    run_dir = checkpoint_path.parent
    saved_args = json.loads(
        (run_dir / "args.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (run_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    saved_mode = saved_args.get("task_mode")
    manifest_mode = manifest.get("task_mode")
    if saved_mode != args.task_mode or manifest_mode != args.task_mode:
        raise ValueError("复评 task_mode 与检查点训练模式或清单不一致")
    required_fields = ("malignant_metadata", "benign_metadata")
    missing_fields = [
        field for field in required_fields if not saved_args.get(field)
    ]
    if missing_fields:
        raise ValueError(
            "检查点参数缺少划分所需字段："
            + ", ".join(missing_fields)
        )
    return checkpoint_path, saved_args, manifest


def evaluate_checkpoint(args, device, canonical_splits):
    """按检查点相邻配置和清单执行确定性复评。"""
    checkpoint_path, saved_args, manifest = load_checkpoint_run_configuration(
        args
    )
    restored = restore_manifest_splits(canonical_splits, manifest)
    model_args = argparse.Namespace(**{**vars(args), **saved_args})
    model = build_model(model_args, device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.task_mode == "overfit":
        dataset = BUSOverfitDataset(
            PROJECT_ROOT,
            samples=restored["train"],
            img_size=model_args.img_size,
        )
    else:
        dataset = MultiModalBreastDataset(
            PROJECT_ROOT,
            split="test",
            img_size=model_args.img_size,
            max_text_len=model_args.max_text_len,
            samples=restored["malignant_test"],
            augment=False,
            ablate_modalities=getattr(model_args, "ablate_modalities", []),
        )
    metrics = evaluate_loader(
        model,
        build_eval_loader(dataset, model_args),
        args.task_mode,
        device,
        split_kind="malignant",
    )
    save_json(Path(args.eval_output), metrics)
    return metrics


def build_overfit_gate_result(overfit_samples, history, val_metrics=None):
    """构造可严格 JSON 序列化的过拟合门禁结论。"""
    final_accuracy = float(history[-1]["train"]["accuracy"])
    final_loss = float(history[-1]["train"]["total_loss"])
    if val_metrics is None:
        prediction_distribution = None
    else:
        prediction_distribution = [
            int(count)
            for count in val_metrics["malignant"]["prediction_distribution"]
        ]
    passed = (
        final_accuracy >= 0.98
        and math.isfinite(final_loss)
        and prediction_distribution is not None
        and len(prediction_distribution) == 4
        and all(count > 0 for count in prediction_distribution)
    )
    if math.isfinite(final_loss):
        serialized_loss = final_loss
    elif math.isnan(final_loss):
        serialized_loss = "nan"
    elif final_loss > 0:
        serialized_loss = "inf"
    else:
        serialized_loss = "-inf"
    return {
        "passed": passed,
        "sample_count": int(overfit_samples),
        "final_accuracy": final_accuracy,
        "final_total_loss": serialized_loss,
        "prediction_distribution": prediction_distribution,
    }


def _raise_overfit_gate_failure(result):
    """依据已构造的门禁结论抛出中文失败信息。"""
    final_loss = result["final_total_loss"]
    loss_text = (
        f"{final_loss:.6f}"
        if isinstance(final_loss, float)
        else final_loss
    )
    raise RuntimeError(
        f"{result['sample_count']} 例过拟合门禁失败："
        f"Accuracy={result['final_accuracy']:.4f}，"
        f"loss={loss_text}，"
        "prediction_distribution="
        f"{result['prediction_distribution']}"
    )


def assert_overfit_gate(overfit_samples, history, val_metrics=None):
    """检查最终训练准确率、损失有限性和四类预测完整性。"""
    result = build_overfit_gate_result(
        overfit_samples,
        history,
        val_metrics,
    )
    if not result["passed"]:
        _raise_overfit_gate_failure(result)
    return result


def enforce_overfit_gate(
    output_dir,
    overfit_samples,
    history,
    val_metrics=None,
):
    """持久化门禁结论，并在失败结论写盘后抛出异常。"""
    result = build_overfit_gate_result(
        overfit_samples,
        history,
        val_metrics,
    )
    save_json(Path(output_dir) / "overfit_gate.json", result)
    if not result["passed"]:
        _raise_overfit_gate_failure(result)
    return result


def validate_overfit_64_prerequisite(args):
    """仅允许已通过匹配 32 例门禁的配置进入 64 例训练。"""
    if args.task_mode != "overfit" or args.overfit_samples != 64:
        return

    output_root = Path(args.output_root)
    if output_root.exists():
        for run_dir in output_root.glob("overfit_*"):
            args_path = run_dir / "args.json"
            gate_path = run_dir / "overfit_gate.json"
            if not args_path.is_file() or not gate_path.is_file():
                continue
            try:
                saved_args = json.loads(args_path.read_text(encoding="utf-8"))
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                gate.get("passed") is True
                and gate.get("sample_count") == 32
                and saved_args.get("seed") == args.seed
                and saved_args.get("pretrained_path") == args.pretrained_path
                and saved_args.get("img_size") == args.img_size
            ):
                return
    raise ValueError(
        "运行 64 例过拟合门禁前，必须先使用相同 seed、预训练权重和图像尺寸通过 32 例门禁"
    )


def _multimodal_dataset(samples, split, args, augment=False):
    """从显式样本清单创建多模态数据集。"""
    return MultiModalBreastDataset(
        PROJECT_ROOT,
        split=split,
        img_size=args.img_size,
        max_text_len=args.max_text_len,
        samples=samples,
        augment=augment,
        ablate_modalities=getattr(args, "ablate_modalities", []),
    )


def _build_evaluation_loaders(splits, args):
    """构造当前任务适用的固定验证与测试加载器。"""
    loaders = {
        "malignant_val": build_eval_loader(
            _multimodal_dataset(
                splits["malignant_val"],
                "val",
                args,
                augment=False,
            ),
            args,
        ),
        "malignant_test": build_eval_loader(
            _multimodal_dataset(
                splits["malignant_test"],
                "test",
                args,
                augment=False,
            ),
            args,
        ),
    }
    if args.task_mode in {"flat5", "dual_head"}:
        loaders["binary_val"] = build_eval_loader(
            _multimodal_dataset(
                splits["binary_val"],
                "val",
                args,
                augment=False,
            ),
            args,
        )
        loaders["binary_test"] = build_eval_loader(
            _multimodal_dataset(
                splits["binary_test"],
                "test",
                args,
                augment=False,
            ),
            args,
        )
    return loaders


def _checkpoint_payload(
    model,
    epoch,
    monitor_metric,
    score,
    malignant_metrics,
    binary_metrics,
):
    """构造可审计的最佳模型检查点。"""
    return {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "monitor_metric": monitor_metric,
        "score": score,
        "malignant_val": malignant_metrics,
        "binary_val": binary_metrics,
    }


def _evaluate_test_sets(model, loaders, args, device):
    """评估固定恶性测试集及适用的二分类测试集。"""
    malignant = evaluate_loader(
        model,
        loaders["malignant_test"],
        args.task_mode,
        device,
        split_kind="malignant",
    )
    binary = None
    if args.task_mode in {"flat5", "dual_head"}:
        binary = evaluate_loader(
            model,
            loaders["binary_test"],
            args.task_mode,
            device,
            split_kind="binary",
        )
    return {"malignant": malignant, "binary": binary}


def main(args):
    """执行可靠基线训练、过拟合门禁或检查点确定性复评。"""
    validate_reliable_configuration(args)
    seed_everything(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    if args.evaluate_checkpoint:
        _, saved_args, _ = load_checkpoint_run_configuration(args)
        canonical_splits = build_fair_splits(
            PROJECT_ROOT,
            saved_args["task_mode"],
            malignant_metadata=saved_args.get(
                "malignant_metadata",
                "metadata.csv",
            ),
            benign_metadata=saved_args.get(
                "benign_metadata",
                "metadata_5class.csv",
            ),
        )
        return evaluate_checkpoint(
            args,
            device,
            canonical_splits,
        )

    validate_overfit_64_prerequisite(args)

    canonical_splits = build_fair_splits(
        PROJECT_ROOT,
        args.task_mode,
        malignant_metadata=args.malignant_metadata,
        benign_metadata=args.benign_metadata,
    )

    if args.task_mode == "overfit":
        splits = {
            "train": select_balanced_subset(
                canonical_splits["train"],
                per_class=args.overfit_samples // 4,
                seed=args.seed,
            )
        }
        augment = False
    else:
        splits = canonical_splits
        augment = args.augment

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        PROJECT_ROOT
        / args.output_root
        / f"{args.task_mode}_{timestamp}"
    )
    manifest = build_split_manifest(args.task_mode, splits, args.seed)
    initialize_run_artifacts(output_dir, args, manifest)

    if args.task_mode == "overfit":
        train_dataset = BUSOverfitDataset(
            PROJECT_ROOT,
            samples=splits["train"],
            img_size=args.img_size,
        )
        evaluation_loaders = {}
    else:
        train_dataset = _multimodal_dataset(
            splits["train"],
            "train",
            args,
            augment=augment,
        )
        evaluation_loaders = _build_evaluation_loaders(
            splits,
            args,
        )

    train_loader = build_train_loader(
        train_dataset,
        args.task_mode,
        args,
    )
    model = build_model(args, device)
    criteria = build_criteria_for_mode(
        args.task_mode,
        train_dataset.samples,
        device,
        label_smoothing=args.label_smoothing,
        flat5_class_weighting=args.flat5_class_weighting,
    )
    if args.task_mode == "overfit":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.0,
        )
    else:
        optimizer = build_optimizer(model, args.lr, args.wd)

    history = []
    best_score = -math.inf
    best_metrics = None
    patience_counter = 0
    last_val_metrics = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criteria,
            args.task_mode,
            device,
            lambda_bm=args.lambda_bm,
            max_grad_norm=args.max_grad_norm,
            accum_steps=args.accum_steps,
        )
        if (
            args.task_mode == "overfit"
            and not math.isfinite(float(train_metrics["total_loss"]))
        ):
            history.append({
                "epoch": epoch,
                "train": train_metrics,
                "malignant_val": None,
                "binary_val": None,
                "score": None,
            })
            enforce_overfit_gate(
                output_dir,
                args.overfit_samples,
                history,
            )
        last_val_metrics = evaluate_loader(
            model,
            (
                train_loader
                if args.task_mode == "overfit"
                else evaluation_loaders["malignant_val"]
            ),
            args.task_mode,
            device,
            split_kind="malignant",
        )
        binary_val_metrics = None
        if args.task_mode in {"flat5", "dual_head"}:
            binary_val_metrics = evaluate_loader(
                model,
                evaluation_loaders["binary_val"],
                args.task_mode,
                device,
                split_kind="binary",
            )
        score = (
            train_metrics["accuracy"]
            if args.task_mode == "overfit"
            else monitor_value(
                last_val_metrics,
                args.monitor_metric,
            )
        )
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "malignant_val": last_val_metrics,
            "binary_val": binary_val_metrics,
            "score": score,
        })
        if score > best_score:
            best_score = score
            patience_counter = 0
            monitor_metric = (
                "train_accuracy"
                if args.task_mode == "overfit"
                else args.monitor_metric
            )
            best_metrics = {
                "epoch": epoch,
                "monitor_metric": monitor_metric,
                "score": score,
                "malignant_val": last_val_metrics,
                "binary_val": binary_val_metrics,
            }
            torch.save(
                _checkpoint_payload(
                    model,
                    epoch,
                    monitor_metric,
                    score,
                    last_val_metrics,
                    binary_val_metrics,
                ),
                output_dir / "best_model.pth",
            )
            print(
                f"第 {epoch} 轮保存最佳模型，"
                f"{monitor_metric}={score:.4f}"
            )
        else:
            patience_counter += 1

        save_training_state(
            output_dir,
            history,
            best_metrics,
        )
        print(
            f"第 {epoch}/{args.epochs} 轮："
            f"训练损失={train_metrics['total_loss']:.6f}，"
            f"训练准确率={train_metrics['accuracy']:.4f}，"
            f"监控值={score:.4f}"
        )
        if (
            args.task_mode != "overfit"
            and patience_counter >= args.patience
        ):
            print(f"连续 {args.patience} 轮未改善，提前停止训练")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": history[-1]["epoch"],
        },
        output_dir / "final_model.pth",
    )

    best_checkpoint = torch.load(
        output_dir / "best_model.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    if args.task_mode == "overfit":
        test_metrics = {
            "malignant": evaluate_loader(
                model,
                train_loader,
                "overfit",
                device,
                split_kind="malignant",
            ),
            "binary": None,
        }
    else:
        test_metrics = _evaluate_test_sets(
            model,
            evaluation_loaders,
            args,
            device,
        )
    save_json(output_dir / "metrics_test.json", test_metrics)
    if args.task_mode == "overfit":
        enforce_overfit_gate(
            output_dir,
            args.overfit_samples,
            history,
            last_val_metrics,
        )
    print(f"训练完成，运行产物已保存到 {output_dir}")
    return test_metrics


if __name__ == "__main__":
    main(build_parser().parse_args())
