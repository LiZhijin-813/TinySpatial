import builtins

import pytest
import torch
from PIL import Image

from code.datasets.dataset import (
    CDFIEvaluationTransform,
    MultiModalBreastDataset,
    PairedAlignedTransform,
    PairedEvaluationTransform,
)
from code.datasets.bus_dataset import BUSOverfitDataset


def test_paired_training_transform_keeps_output_size():
    bus = Image.new("RGB", (300, 300), color=(128, 0, 0))
    swe = Image.new("RGB", (300, 300), color=(0, 0, 128))
    bus_out, swe_out = PairedAlignedTransform(img_size=224)(bus, swe)
    assert bus_out.size == swe_out.size == (224, 224)


def test_evaluation_transforms_are_deterministic():
    image = Image.new("RGB", (300, 260), color=(120, 20, 80))
    paired = PairedEvaluationTransform(img_size=224)
    first_bus, first_swe = paired(image, image)
    second_bus, second_swe = paired(image, image)
    assert first_bus.tobytes() == second_bus.tobytes()
    assert first_swe.tobytes() == second_swe.tobytes()
    cdfi = CDFIEvaluationTransform(img_size=224)
    assert cdfi(image).tobytes() == cdfi(image).tobytes()


def test_dataset_returns_explicit_labels_and_fixed_tensors(
    tiny_multimodal_root,
    fake_tokenizer,
):
    root, samples = tiny_multimodal_root
    dataset = MultiModalBreastDataset(
        root_dir=str(root),
        split="val",
        img_size=224,
        max_text_len=16,
        samples=samples,
        augment=False,
        tokenizer=fake_tokenizer,
    )
    first = dataset[0]
    second = dataset[0]
    assert first["case_id"] == "case-malignant"
    assert first["bus_img"].shape == (1, 224, 224)
    assert first["swe_img"].shape == (3, 224, 224)
    assert first["cdfi_img"].shape == (3, 224, 224)
    assert first["input_ids"].shape == (16,)
    assert first["class_label"] == 2
    assert first["malignancy_label"] == 1
    assert first["subtype_label"] == 2
    assert torch.equal(first["bus_img"], second["bus_img"])
    assert torch.equal(first["swe_img"], second["swe_img"])
    assert torch.equal(first["cdfi_img"], second["cdfi_img"])

    benign = dataset[1]
    assert benign["class_label"] == 4
    assert benign["malignancy_label"] == 0
    assert benign["subtype_label"] == -1


def test_augment_defaults_to_train_only(tiny_multimodal_root, fake_tokenizer):
    root, samples = tiny_multimodal_root
    train = MultiModalBreastDataset(
        str(root), split="train", samples=samples, tokenizer=fake_tokenizer
    )
    val = MultiModalBreastDataset(
        str(root), split="val", samples=samples, tokenizer=fake_tokenizer
    )
    assert train.augment is True
    assert val.augment is False


def test_bus_overfit_dataset_loads_no_other_modalities(tiny_multimodal_root):
    root, samples = tiny_multimodal_root
    dataset = BUSOverfitDataset(
        root_dir=str(root),
        samples=[samples[0]],
        img_size=224,
    )
    first = dataset[0]
    second = dataset[0]
    assert set(first) == {"case_id", "bus_img", "subtype_label"}
    assert first["bus_img"].shape == (1, 224, 224)
    assert first["subtype_label"] == 2
    assert torch.equal(first["bus_img"], second["bus_img"])


def test_benign_labels_are_normalized_for_explicit_and_metadata_samples(
    tiny_multimodal_root,
    fake_tokenizer,
):
    """显式样本和旧 metadata 的良性标签都应统一为三类标签语义。"""
    root, _ = tiny_multimodal_root
    explicit = MultiModalBreastDataset(
        root_dir=str(root),
        split="val",
        max_text_len=16,
        samples=[{"case_id": "case-benign", "subtype_label": -1}],
        augment=False,
        tokenizer=fake_tokenizer,
    )[0]
    assert explicit["class_label"] == 4
    assert explicit["malignancy_label"] == 0
    assert explicit["subtype_label"] == -1

    (root / "data" / "metadata.csv").write_text(
        "case_id,subtype_label,split\ncase-benign,4,val\n",
        encoding="utf-8",
    )
    metadata = MultiModalBreastDataset(
        root_dir=str(root),
        split="val",
        max_text_len=16,
        augment=False,
        tokenizer=fake_tokenizer,
    )[0]
    assert metadata["class_label"] == 4
    assert metadata["malignancy_label"] == 0
    assert metadata["subtype_label"] == -1


def test_bus_overfit_dataset_reads_after_other_modalities_are_removed(
    tiny_multimodal_root,
):
    """删除目标病例的其他模态后，BUS-only 数据集仍应成功读取 BUS。"""
    root, samples = tiny_multimodal_root
    case_id = samples[0]["case_id"]
    (root / "data" / "images" / "SWE" / f"{case_id}.jpg").unlink()
    (root / "data" / "images" / "CDFI" / f"{case_id}.jpg").unlink()
    (root / "data" / "texts" / f"{case_id}.json").unlink()

    sample = BUSOverfitDataset(
        root_dir=str(root),
        samples=[samples[0]],
        img_size=224,
    )[0]
    assert sample["case_id"] == case_id
    assert sample["bus_img"].shape == (1, 224, 224)


def test_clinical_text_json_is_opened_with_utf8(
    tiny_multimodal_root,
    fake_tokenizer,
    monkeypatch,
):
    """临床文本 JSON 必须显式使用 UTF-8 打开。"""
    root, samples = tiny_multimodal_root
    original_open = builtins.open
    encodings = []

    def spy_open(file, mode="r", *args, **kwargs):
        if str(file).endswith(".json"):
            encodings.append(kwargs.get("encoding"))
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    dataset = MultiModalBreastDataset(
        root_dir=str(root),
        split="val",
        max_text_len=16,
        samples=[samples[0]],
        augment=False,
        tokenizer=fake_tokenizer,
    )
    dataset[0]
    assert encodings == ["utf-8"]


def test_evaluation_transform_rejects_mismatched_paired_sizes():
    """BUS 和 SWE 原始尺寸不一致时，评估变换必须拒绝伪对齐。"""
    bus = Image.new("RGB", (300, 300), color=(128, 0, 0))
    swe = Image.new("RGB", (280, 300), color=(0, 0, 128))
    with pytest.raises(ValueError):
        PairedEvaluationTransform(img_size=224)(bus, swe)
