import json
from pathlib import Path

import pytest
import torch
from PIL import Image


class FakeTokenizer:
    cls_token_id = 101

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        values = [101, 200 + len(text), 102]
        values = values[:max_length] + [0] * max(0, max_length - len(values))
        mask = [int(value != 0) for value in values]
        return {
            "input_ids": torch.tensor([values], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def tiny_multimodal_root(tmp_path: Path):
    for modality in ("BUS", "SWE", "CDFI"):
        (tmp_path / "data" / "images" / modality).mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "texts").mkdir(parents=True)

    samples = [
        {
            "case_id": "case-malignant",
            "class_label": 2,
            "malignancy_label": 1,
            "subtype_label": 2,
            "split": "val",
            "source": "malignant",
        },
        {
            "case_id": "case-benign",
            "class_label": 4,
            "malignancy_label": 0,
            "subtype_label": -1,
            "split": "val",
            "source": "five_class",
        },
    ]
    for index, sample in enumerate(samples):
        case_id = sample["case_id"]
        Image.new("RGB", (280, 240), (30 + index, 60, 90)).save(
            tmp_path / "data" / "images" / "BUS" / f"{case_id}.jpg"
        )
        Image.new("RGB", (280, 240), (90, 30 + index, 60)).save(
            tmp_path / "data" / "images" / "SWE" / f"{case_id}.jpg"
        )
        Image.new("RGB", (260, 250), (60, 90, 30 + index)).save(
            tmp_path / "data" / "images" / "CDFI" / f"{case_id}.jpg"
        )
        with (tmp_path / "data" / "texts" / f"{case_id}.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump({"raw_text": f"病例 {case_id}"}, handle, ensure_ascii=False)
    return tmp_path, samples
