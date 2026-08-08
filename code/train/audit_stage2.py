"""恢复 flat5 运行并生成恶性病例级错误审计产物。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

import torch

from code.datasets.dataset import (
    MultiModalBreastDataset,
    VALID_ABLATION_MODALITIES,
)
from code.train.case_audit import (
    build_audit_summary,
    build_case_records,
    validate_output_directory,
    write_audit_outputs,
)
from code.train.train_stage2 import (
    build_eval_loader,
    build_fair_splits,
    build_model,
    restore_manifest_splits,
)


class _ChineseArgumentParser(argparse.ArgumentParser):
    """将参数解析错误统一为中文提示。"""

    def error(self, message):
        self.exit(2, "参数错误，请检查必需参数及参数取值。\n")


def _positive_integer(value):
    """解析严格为正的批次大小。"""
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("批次大小必须是正整数") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("批次大小必须是正整数")
    return number


def build_parser():
    """构造 flat5 病例审计命令行解析器。"""
    parser = _ChineseArgumentParser(description="flat5 恶性病例错误审计")
    parser.add_argument("--run_dir", type=Path, required=True, help="已完成运行目录")
    parser.add_argument("--output_dir", type=Path, help="审计产物目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--batch_size", type=_positive_integer, help="审计批次大小")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖非空审计目录")
    parser.add_argument(
        "--ablate_modalities",
        nargs="*",
        choices=VALID_ABLATION_MODALITIES,
        default=[],
        help="实验时屏蔽的模态列表",
    )
    return parser


def load_flat5_audit_run(run_dir):
    """读取并校验 flat5 审计所需的四个只读运行产物。"""
    run_path = Path(run_dir)
    required = {
        "参数文件": run_path / "args.json",
        "划分清单": run_path / "split_manifest.json",
        "最佳模型": run_path / "best_model.pth",
        "测试指标": run_path / "metrics_test.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError("运行目录缺少必需文件：" + "、".join(missing))

    saved_args = _read_json(required["参数文件"], "参数文件")
    manifest = _read_json(required["划分清单"], "划分清单")
    metrics = _read_json(required["测试指标"], "测试指标")
    if not isinstance(saved_args, dict) or not isinstance(manifest, dict):
        raise ValueError("参数文件和划分清单必须是对象")
    if saved_args.get("task_mode") != "flat5" or manifest.get("task_mode") != "flat5":
        raise ValueError("保存的运行和划分清单必须均为 flat5")
    batch_size = saved_args.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("保存的批次大小必须是正整数")

    splits = manifest.get("splits")
    malignant_test = splits.get("malignant_test") if isinstance(splits, dict) else None
    if (
        not isinstance(malignant_test, list)
        or not malignant_test
        or any(not isinstance(case_id, str) or not case_id.strip() for case_id in malignant_test)
        or len(set(malignant_test)) != len(malignant_test)
    ):
        raise ValueError("划分清单中的恶性测试病例必须非空且不重复")
    if not isinstance(metrics, dict):
        raise ValueError("测试指标文件必须是对象")
    return argparse.Namespace(**saved_args), manifest, required["最佳模型"], metrics


def collect_flat5_audit_inputs(model, loader, device, dataset):
    """按加载器顺序收集 flat5 审计记录所需的最小推理结果。"""
    case_ids = []
    subtype_labels = []
    logits = []
    modality_exists = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_case_ids = _batch_case_ids(batch)
            try:
                labels = batch["subtype_label"]
                outputs = model(
                    batch["bus_img"].to(device),
                    batch["swe_img"].to(device),
                    batch["cdfi_img"].to(device),
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                batch_logits = outputs["class_logits"]
            except (KeyError, AttributeError, TypeError) as error:
                raise ValueError("审计推理批次缺少必要的多模态输入或分类分数") from error
            if not isinstance(labels, torch.Tensor) or not isinstance(batch_logits, torch.Tensor):
                raise ValueError("审计推理标签和分类分数必须为张量")
            if labels.ndim != 1 or batch_logits.ndim != 2 or batch_logits.shape[0] != len(batch_case_ids):
                raise ValueError("审计推理批次的病例、标签和分类分数长度不一致")
            case_ids.extend(batch_case_ids)
            subtype_labels.extend(int(value) for value in labels.detach().cpu().tolist())
            logits.append(batch_logits.detach().cpu())
            modality_exists.extend(_modality_exists(dataset, case_id) for case_id in batch_case_ids)

    if not logits:
        raise ValueError("审计数据加载器未产生任何批次")
    expected_case_ids = [sample.get("case_id") for sample in dataset.samples]
    if case_ids != expected_case_ids:
        raise ValueError("审计加载顺序与恢复的恶性测试病例顺序不一致")
    return case_ids, subtype_labels, torch.cat(logits, dim=0), modality_exists


def run_case_audit(args):
    """恢复保存的 flat5 运行，生成独立且可复现的病例审计目录。"""
    _validate_requested_batch_size(args.batch_size)
    saved_args, manifest, checkpoint_path, saved_metrics = load_flat5_audit_run(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.run_dir) / "case_audit"
    output_dir = validate_output_directory(output_dir, args.overwrite)
    device = _resolve_device(args.device)
    canonical_splits = build_fair_splits(
        PROJECT_ROOT,
        "flat5",
        malignant_metadata=getattr(saved_args, "malignant_metadata", "metadata.csv"),
        benign_metadata=getattr(saved_args, "benign_metadata", "metadata_5class.csv"),
    )
    restored_splits = restore_manifest_splits(canonical_splits, manifest)
    malignant_samples = restored_splits.get("malignant_test")
    if not malignant_samples:
        raise ValueError("无法从划分清单恢复恶性测试病例")

    model = build_model(saved_args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("最佳模型文件缺少模型权重")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = MultiModalBreastDataset(
        PROJECT_ROOT,
        split="test",
        img_size=saved_args.img_size,
        max_text_len=saved_args.max_text_len,
        samples=malignant_samples,
        augment=False,
        ablate_modalities=getattr(args, "ablate_modalities", []),
    )
    loader_args = argparse.Namespace(**vars(saved_args))
    loader_args.batch_size = args.batch_size or saved_args.batch_size
    loader_args.num_workers = getattr(saved_args, "num_workers", 0)
    inputs = collect_flat5_audit_inputs(model, build_eval_loader(dataset, loader_args), device, dataset)
    records = build_case_records(*inputs)
    summary = build_audit_summary(
        records,
        saved_metrics,
        ablate_modalities=getattr(args, "ablate_modalities", []),
    )
    write_audit_outputs(output_dir, records, summary, overwrite=args.overwrite)
    print(f"病例审计完成：共 {len(records)} 例，输出目录：{output_dir}")
    return output_dir


def _read_json(path, name):
    """读取对象形式的 UTF-8 JSON 运行产物。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name}无法读取为有效 JSON") from error


def _validate_requested_batch_size(batch_size):
    """校验调用方直接传入的可选审计批次大小。"""
    if batch_size is None:
        return
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("审计批次大小必须是正整数")


def _resolve_device(device_name):
    """解析设备并在 CUDA 不可用时给出中文错误。"""
    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError) as error:
        raise ValueError("推理设备参数无效") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求了 CUDA 设备，但当前环境不可用")
    return device


def _batch_case_ids(batch):
    """校验并复制一个审计批次中的病例编号。"""
    try:
        case_ids = list(batch["case_id"])
    except (KeyError, TypeError) as error:
        raise ValueError("审计推理批次缺少病例编号") from error
    if not case_ids or any(not isinstance(case_id, str) or not case_id.strip() for case_id in case_ids):
        raise ValueError("审计推理批次包含无效病例编号")
    return case_ids


def _modality_exists(dataset, case_id):
    """根据数据集目录属性构造四种模态的存在性，而不读取原始内容。"""
    try:
        return {
            "bus_exists": (Path(dataset.bus_dir) / f"{case_id}.jpg").is_file(),
            "swe_exists": (Path(dataset.swe_dir) / f"{case_id}.jpg").is_file(),
            "cdfi_exists": (Path(dataset.cdfi_dir) / f"{case_id}.jpg").is_file(),
            "text_exists": (Path(dataset.texts_dir) / f"{case_id}.json").is_file(),
        }
    except AttributeError as error:
        raise ValueError("审计数据集缺少模态目录属性") from error


if __name__ == "__main__":
    run_case_audit(build_parser().parse_args())
