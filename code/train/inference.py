"""推理脚本：单例预测 + 批量评估。

支持两种推理模式：
    - single: 对单个病例进行亚型预测，输出概率分布
    - batch: 对整个数据集进行批量推理与评估，输出完整临床评估报告
"""
import os
import sys
import argparse
import json

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.utils.seed import seed_everything
from code.datasets.dataset import MultiModalBreastDataset
from code.models.stage2.subtyping_model import SECSubtypingModel
from code.utils.evaluation import evaluate_predictions, print_evaluation

SUBTYPE_NAMES_4 = ["Luminal A", "Luminal B", "HER2+", "TNBC"]
SUBTYPE_NAMES_5 = ["Luminal A", "Luminal B", "HER2+", "TNBC", "Benign"]


def task_predictions_and_probabilities(outputs, task_mode):
    """按任务模式将结构化 logits 转换为预测类别和概率。"""
    if task_mode in {"flat4", "flat5"}:
        probabilities = torch.softmax(outputs["class_logits"], dim=1)
        return probabilities.argmax(dim=1), probabilities

    if task_mode == "dual_head":
        malignancy_probabilities = torch.softmax(outputs["malignancy_logits"], dim=1)
        subtype_probabilities = torch.softmax(outputs["subtype_logits"], dim=1)
        predictions = subtype_probabilities.argmax(dim=1)
        benign_predictions = torch.full_like(predictions, 4)
        predictions = torch.where(
            outputs["malignancy_logits"].argmax(dim=1) == 0,
            benign_predictions,
            predictions,
        )
        probabilities = torch.cat(
            [
                malignancy_probabilities[:, 1:2] * subtype_probabilities,
                malignancy_probabilities[:, 0:1],
            ],
            dim=1,
        )
        return predictions, probabilities

    raise ValueError(f"不支持的任务模式: {task_mode}")


def _extract_state_dict(checkpoint):
    """从常见检查点结构中提取参数字典。"""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("检查点必须是参数字典或包含参数字典的检查点")
    return state_dict


def load_stage2_checkpoint(model, checkpoint_path, device):
    """加载结构化 Stage 2 检查点，并拒绝不兼容的分类头。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    keys = tuple(state_dict)

    if any(key.startswith("mlp_head.") for key in keys):
        raise ValueError("检查点包含旧版 mlp_head 分类头，无法用于结构化任务接口")
    if not any(key.startswith("task_heads.") for key in keys):
        raise ValueError("检查点缺少任务头参数，无法用于结构化任务接口")

    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise ValueError("检查点参数形状不匹配，无法加载结构化任务接口") from exc

    if any(key.startswith("task_heads.") for key in incompatible.missing_keys):
        raise ValueError("当前模型缺少检查点要求的任务头参数")
    if any(key.startswith("mlp_head.") for key in incompatible.unexpected_keys):
        raise ValueError("检查点包含旧版 mlp_head 分类头，无法用于结构化任务接口")

    print(f"已加载结构化 Stage 2 检查点: {checkpoint_path}")


@torch.no_grad()
def predict_batch(model, dataloader, device, task_mode):
    """对数据集执行批量推理，返回预测结果与标签。

    Args:
        model: SECSubtypingModel 模型
        dataloader: 数据加载器
        device: 计算设备
        task_mode: 任务模式

    Returns:
        preds: (N,) 预测标签数组
        labels: (N,) 真实标签数组
        probs: (N, C) 预测概率数组
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(dataloader, desc="推理中"):
        bus_img = batch["bus_img"].to(device)
        swe_img = batch["swe_img"].to(device)
        cdfi_img = batch["cdfi_img"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(bus_img, swe_img, cdfi_img, input_ids, attention_mask)
        predictions, probabilities = task_predictions_and_probabilities(outputs, task_mode)
        label_key = "subtype_label" if task_mode == "flat4" else "class_label"

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch[label_key].numpy())
        all_probs.extend(probabilities.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def predict_single(
    model,
    bus_path,
    swe_path,
    cdfi_path,
    text_path,
    device,
    task_mode,
    img_size=224,
    max_text_len=128,
    subtype_names=None,
):
    """对单个病例进行亚型预测。

    Args:
        model: SECSubtypingModel 模型
        bus_path: BUS 图像路径
        swe_path: SWE 图像路径
        cdfi_path: CDFI 图像路径
        text_path: 临床文本 JSON 路径
        device: 计算设备
        task_mode: 任务模式
        img_size: 输入图像尺寸，默认 224
        max_text_len: 文本最大长度，默认 128

    Returns:
        包含 predicted_subtype, predicted_label, probabilities 的结果字典
    """
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoTokenizer

    # 配置 HuggingFace 镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DOWNLOAD_ENDPOINT'] = 'https://hf-mirror.com'

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    normalize_3ch = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((img_size, img_size))

    # 加载 BUS 图像并转为灰度单通道
    bus_img = Image.open(bus_path).convert("L")
    bus_tensor = normalize(to_tensor(resize(bus_img)))  # (1, H, W)

    # 加载 SWE 图像
    swe_img = Image.open(swe_path).convert("RGB")
    swe_tensor = normalize_3ch(to_tensor(resize(swe_img)))  # (3, H, W)

    # 加载 CDFI 图像
    cdfi_img = Image.open(cdfi_path).convert("RGB")
    cdfi_tensor = normalize_3ch(to_tensor(resize(cdfi_img)))  # (3, H, W)

    # 加载临床文本并进行 tokenization
    with open(text_path) as f:
        text_data = json.load(f)
    raw_text = text_data.get("raw_text", "")
    encoding = tokenizer(raw_text, max_length=max_text_len, padding="max_length", truncation=True, return_tensors="pt")

    # 添加批次维度
    bus_tensor = bus_tensor.unsqueeze(0).to(device)
    swe_tensor = swe_tensor.unsqueeze(0).to(device)
    cdfi_tensor = cdfi_tensor.unsqueeze(0).to(device)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 推理
    with torch.no_grad():
        outputs = model(bus_tensor, swe_tensor, cdfi_tensor, input_ids, attention_mask)
        predictions, probabilities = task_predictions_and_probabilities(outputs, task_mode)
        probs = probabilities.cpu().squeeze(0)

    pred_class = predictions.item()
    names = subtype_names or (
        SUBTYPE_NAMES_4 if task_mode == "flat4" else SUBTYPE_NAMES_5
    )
    result = {
        "predicted_subtype": names[pred_class],
        "predicted_label": pred_class,
        "probabilities": {name: float(probs[i]) for i, name in enumerate(names)},
    }
    return result


def main(args):
    """推理主函数。"""
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载模型
    subtype_names = (
        SUBTYPE_NAMES_4 if args.task_mode == "flat4" else SUBTYPE_NAMES_5
    )
    model = SECSubtypingModel(
        pretrained_path=None,
        task_mode=args.task_mode,
        img_size=args.img_size,
    ).to(device)

    if args.checkpoint:
        load_stage2_checkpoint(model, args.checkpoint, device)

    if args.mode == "single":
        # 单例预测模式
        result = predict_single(
            model, args.bus_path, args.swe_path, args.cdfi_path, args.text_path, device,
            args.task_mode, img_size=args.img_size, subtype_names=subtype_names,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == "batch":
        # 批量评估模式
        dataset = MultiModalBreastDataset(
            root_dir=PROJECT_ROOT, split=args.split, img_size=args.img_size,
            metadata_file=args.metadata_file,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        preds, labels, probs = predict_batch(model, loader, device, args.task_mode)

        # 临床评估
        results = evaluate_predictions(
            labels,
            preds,
            class_names=subtype_names,
            save_dir=args.output_dir,
        )
        print_evaluation(results)

        # 保存详细评估结果
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(
                os.path.join(args.output_dir, "evaluation_results.json"),
                "w",
                encoding="utf-8",
            ) as f:
                # NumPy 类型 JSON 序列化转换器
                def convert(obj):
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                json.dump(results, f, indent=2, default=convert, ensure_ascii=False)
            print(f"评估结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="推理与评估")
    parser.add_argument("--mode", type=str, default="batch", choices=["single", "batch"],
                        help="推理模式: single(单例) 或 batch(批量)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练好的模型检查点路径")
    parser.add_argument(
        "--task_mode",
        type=str,
        default="flat4",
        choices=["flat4", "flat5", "dual_head"],
        help="任务模式：flat4、flat5 或 dual_head",
    )
    parser.add_argument("--metadata_file", type=str, default="metadata.csv",
                        help="元数据文件名 (4分类用 metadata.csv, 5分类用 metadata_5class.csv)")
    parser.add_argument("--split", type=str, default="test",
                        help="批量模式下的数据集划分，默认 test")
    parser.add_argument("--img_size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--output_dir", type=str, default=None, help="结果输出目录")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")

    # 单例模式路径参数
    parser.add_argument("--bus_path", type=str, default=None, help="BUS 图像路径 (单例模式)")
    parser.add_argument("--swe_path", type=str, default=None, help="SWE 图像路径 (单例模式)")
    parser.add_argument("--cdfi_path", type=str, default=None, help="CDFI 图像路径 (单例模式)")
    parser.add_argument("--text_path", type=str, default=None, help="临床文本 JSON 路径 (单例模式)")

    args = parser.parse_args()
    main(args)
