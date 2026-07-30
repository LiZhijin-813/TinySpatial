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
