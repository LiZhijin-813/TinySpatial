"""多模态乳腺超声数据集模块。

实现 MultiModalBreastDataset，通过 case_id 索引并关联 BUS、SWE、CDFI
三模态影像与临床文本报告，返回统一的张量字典。

核心特性：
    - 空间对齐增强：BUS 与 SWE 共享随机参数执行相同几何变换
    - 独立适配增强：CDFI 允许弹性形变与色彩抖动
    - AutoTokenizer 集成：对临床文本进行定长 tokenization
"""
import os

# 配置 HuggingFace 镜像，用于无法直接访问 huggingface.co 的环境
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_ENDPOINT'] = 'https://hf-mirror.com'

import json
import csv
import math
import random
from typing import Dict, Optional, Callable, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from transformers import AutoTokenizer


class PairedAlignedTransform:
    """BUS 与 SWE 空间对齐增强。

    对 BUS 和 SWE 图像执行共享随机参数的仿射变换（随机裁剪、水平翻转、旋转），
    确保两模态特征空间的相对坐标绝对静止，维持解剖结构与硬度分布的像素级对应。
    """

    def __init__(self, img_size: int = 224):
        self.img_size = img_size

    def __call__(self, bus_img: Image.Image, swe_img: Image.Image):
        # 先将短边缩放至至少 img_size，确保图像足够大以支持 RandomCrop
        w, h = bus_img.size
        scale = max(self.img_size / w, self.img_size / h)
        if scale > 1.0:
            new_w = max(math.ceil(w * scale), self.img_size)
            new_h = max(math.ceil(h * scale), self.img_size)
            bus_img = TF.resize(bus_img, [new_h, new_w])
            swe_img = TF.resize(swe_img, [new_h, new_w])

        # 共享随机裁剪参数：确保裁剪区域完全一致
        i, j, h, w = transforms.RandomCrop.get_params(
            bus_img, output_size=(self.img_size, self.img_size)
        )
        bus_img = TF.crop(bus_img, i, j, h, w)
        swe_img = TF.crop(swe_img, i, j, h, w)

        # 共享随机水平翻转
        if random.random() > 0.5:
            bus_img = TF.hflip(bus_img)
            swe_img = TF.hflip(swe_img)

        # 共享随机旋转（±15°），fill=0 用黑色填充边缘
        angle = random.uniform(-15, 15)
        bus_img = TF.rotate(bus_img, angle, fill=0)
        swe_img = TF.rotate(swe_img, angle, fill=0)

        return bus_img, swe_img


class CDFIIndependentTransform:
    """CDFI 独立适配增强。

    CDFI 反映的是血流分布，不一定与组织物理边界实时严格对应。
    因此允许独立的弹性形变与色彩抖动，模拟临床探头按压导致的血流变形。
    """

    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # 色彩抖动：模拟不同设备/参数下的血流伪彩差异
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ])

    def __call__(self, cdfi_img: Image.Image):
        return self.transform(cdfi_img)


class MultiModalBreastDataset(Dataset):
    """多模态乳腺超声数据集。

    通过 case_id 从 metadata.csv 索引并关联各模态文件，
    __getitem__ 返回如下结构的字典：

        bus_img:         Tensor (1, H, W) — 灰度 BUS 解剖图
        swe_img:         Tensor (3, H, W) — SWE 剪切波硬度伪彩图
        cdfi_img:        Tensor (3, H, W) — CDFI 彩色多普勒血流图
        input_ids:       Tensor (max_len,) — 临床文本 token ID
        attention_mask:  Tensor (max_len,) — 注意力掩码
        subtype_label:   int — 分子亚型类别 (0-3)

    Attributes:
        SUBTYPE_MAP: 亚型编号到名称的映射
    """

    SUBTYPE_MAP = {0: "Luminal A", 1: "Luminal B", 2: "HER2+", 3: "TNBC", 4: "Benign"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 224,
        max_text_len: int = 128,
        tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        transform_paired: Optional[Callable] = None,
        transform_cdfi: Optional[Callable] = None,
        metadata_file: str = "metadata.csv",
    ):
        """
        Args:
            root_dir: 项目根目录路径
            split: 数据划分，可选 "train"/"val"/"test"
            img_size: 输出图像尺寸，默认 224×224
            max_text_len: 文本 tokenization 最大长度，默认 128
            tokenizer_name: HuggingFace tokenizer 名称
            transform_paired: 自定义 BUS-SWE 对齐增强，默认使用 PairedAlignedTransform
            transform_cdfi: 自定义 CDFI 独立增强，默认使用 CDFIIndependentTransform
            metadata_file: 元数据文件名，默认 "metadata.csv"（4分类），五分类使用 "metadata_5class.csv"
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.max_text_len = max_text_len

        # 各模态数据目录
        self.data_dir = os.path.join(root_dir, "data")
        self.metadata_path = os.path.join(self.data_dir, metadata_file)
        self.bus_dir = os.path.join(self.data_dir, "images", "BUS")
        self.swe_dir = os.path.join(self.data_dir, "images", "SWE")
        self.cdfi_dir = os.path.join(self.data_dir, "images", "CDFI")
        self.texts_dir = os.path.join(self.data_dir, "texts")

        # 加载 BioClinicalBERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # 按 split 加载元数据
        self.samples = self._load_metadata()

        # 初始化增强策略
        self.transform_paired = transform_paired or PairedAlignedTransform(img_size)
        self.transform_cdfi = transform_cdfi or CDFIIndependentTransform(img_size)

        # 几何增强后的归一化与张量转换
        self.bus_normalize = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # BUS 转为单通道灰度
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.swe_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.cdfi_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_metadata(self) -> List[Dict]:
        """从 metadata.csv 加载当前 split 的样本列表。"""
        samples = []
        with open(self.metadata_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == self.split:
                    samples.append({
                        "case_id": row["case_id"],
                        "subtype_label": int(row["subtype_label"]),
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本的多模态数据。

        Args:
            idx: 样本索引

        Returns:
            包含 bus_img, swe_img, cdfi_img, input_ids, attention_mask, subtype_label 的字典
        """
        sample = self.samples[idx]
        case_id = sample["case_id"]

        # 加载三模态图像
        bus_img = Image.open(os.path.join(self.bus_dir, f"{case_id}.jpg")).convert("RGB")
        swe_img = Image.open(os.path.join(self.swe_dir, f"{case_id}.jpg")).convert("RGB")
        cdfi_img = Image.open(os.path.join(self.cdfi_dir, f"{case_id}.jpg")).convert("RGB")

        # BUS 与 SWE 执行空间对齐增强
        bus_img, swe_img = self.transform_paired(bus_img, swe_img)

        # CDFI 执行独立适配增强
        cdfi_img = self.transform_cdfi(cdfi_img)

        # 归一化并转为张量
        bus_tensor = self.bus_normalize(bus_img)      # (1, H, W)
        swe_tensor = self.swe_normalize(swe_img)      # (3, H, W)
        cdfi_tensor = self.cdfi_normalize(cdfi_img)   # (3, H, W)

        # 加载临床文本并进行 tokenization
        text_path = os.path.join(self.texts_dir, f"{case_id}.json")
        with open(text_path, "r") as f:
            text_data = json.load(f)
        raw_text = text_data.get("raw_text", "")

        encoding = self.tokenizer(
            raw_text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "bus_img": bus_tensor,
            "swe_img": swe_tensor,
            "cdfi_img": cdfi_tensor,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "subtype_label": sample["subtype_label"],
        }
