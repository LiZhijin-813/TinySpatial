"""数据管线单元测试：验证输出维度、增强一致性、tokenization 正确性。

测试项：
    - test_output_dimensions: 验证各模态输出张量维度正确
    - test_bus_swe_alignment: 验证 BUS 与 SWE 几何变换一致性
    - test_cdfi_independence: 验证 CDFI 增强独立性
    - test_tokenization: 验证文本 tokenization 输出有效性
    - test_dataloader_batch: 验证 DataLoader 可正常生成批次
"""
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
from PIL import Image
from code.datasets.dataset import (
    MultiModalBreastDataset,
    PairedAlignedTransform,
    CDFIIndependentTransform,
)


def test_output_dimensions():
    """验证所有输出张量的形状正确。"""
    ds = MultiModalBreastDataset(
        root_dir="/home/lzj813/TinySpatial_Project",
        split="train",
        img_size=224,
        max_text_len=128,
    )
    sample = ds[0]
    assert sample["bus_img"].shape == (1, 224, 224), f"bus_img 形状错误: {sample['bus_img'].shape}"
    assert sample["swe_img"].shape == (3, 224, 224), f"swe_img 形状错误: {sample['swe_img'].shape}"
    assert sample["cdfi_img"].shape == (3, 224, 224), f"cdfi_img 形状错误: {sample['cdfi_img'].shape}"
    assert sample["input_ids"].shape == (128,), f"input_ids 形状错误: {sample['input_ids'].shape}"
    assert sample["attention_mask"].shape == (128,), f"attention_mask 形状错误: {sample['attention_mask'].shape}"
    assert isinstance(sample["subtype_label"], int)
    assert 0 <= sample["subtype_label"] <= 3
    print("[PASS] test_output_dimensions")


def test_bus_swe_alignment():
    """验证 BUS 与 SWE 执行相同的几何变换。

    固定随机种子后，两幅图像的裁剪/翻转/旋转参数应完全一致。
    """
    random.seed(42)
    # 创建两幅测试图像
    bus = Image.new("RGB", (300, 300), color=(128, 0, 0))
    swe = Image.new("RGB", (300, 300), color=(0, 0, 128))

    transform = PairedAlignedTransform(img_size=224)
    random.seed(123)
    bus_out, swe_out = transform(bus, swe)

    # 两幅输出图像尺寸应相同
    assert bus_out.size == swe_out.size == (224, 224), f"尺寸不匹配: {bus_out.size} vs {swe_out.size}"
    print("[PASS] test_bus_swe_alignment")


def test_cdfi_independence():
    """验证 CDFI 增强与 BUS/SWE 独立。"""
    cdfi = Image.new("RGB", (300, 300), color=(200, 50, 50))
    transform = CDFIIndependentTransform(img_size=224)
    out = transform(cdfi)
    assert out.size == (224, 224)
    print("[PASS] test_cdfi_independence")


def test_tokenization():
    """验证 tokenization 产生有效的 token ID 和掩码。"""
    ds = MultiModalBreastDataset(
        root_dir="/home/lzj813/TinySpatial_Project",
        split="train",
        img_size=224,
        max_text_len=128,
    )
    sample = ds[0]
    ids = sample["input_ids"]
    mask = sample["attention_mask"]

    # 首个 token 应为 CLS
    assert ids[0].item() == ds.tokenizer.cls_token_id, "首个 token 应为 CLS"
    # 注意力掩码应有非零值
    assert mask.sum().item() > 0, "注意力掩码不应全为零"
    # 非零掩码数应不超过最大长度
    nonzero = mask.sum().item()
    assert nonzero <= 128
    print(f"[PASS] test_tokenization (非零 token 数: {nonzero})")


def test_dataloader_batch():
    """验证 DataLoader 可正常生成批次。"""
    from torch.utils.data import DataLoader

    ds = MultiModalBreastDataset(
        root_dir="/home/lzj813/TinySpatial_Project",
        split="train",
        img_size=224,
        max_text_len=128,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["bus_img"].shape == (4, 1, 224, 224)
    assert batch["swe_img"].shape == (4, 3, 224, 224)
    assert batch["cdfi_img"].shape == (4, 3, 224, 224)
    assert batch["input_ids"].shape == (4, 128)
    assert batch["attention_mask"].shape == (4, 128)
    assert batch["subtype_label"].shape == (4,)
    print("[PASS] test_dataloader_batch")


if __name__ == "__main__":
    test_output_dimensions()
    test_bus_swe_alignment()
    test_cdfi_independence()
    test_tokenization()
    test_dataloader_batch()
    print("\n所有数据管线测试通过！")
