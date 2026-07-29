# Stage 2 可靠多任务基线实施计划

> **面向智能执行者：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项执行本计划。所有步骤使用复选框（`- [ ]`）跟踪。

**目标：** 建立可审计、可复现、能通过小样本过拟合门禁的 Stage 2 训练管线，并在完全一致的恶性验证集/测试集上公平比较 `flat4`、`flat5` 与 `dual_head`。

**架构：** 数据层负责生成固定的实验划分与显式任务标签，模型层共享现有四模态特征提取器并按任务模式挂接平坦或双头分类器，训练层统一处理损失、评估、模型选择和轻量实验产物。先通过 BUS-only 的 32/64 例过拟合门禁，再运行 B0/B1/B2；第一轮基线关闭 CAM、特征多样性正则和标签平滑。

**技术栈：** Python 3、PyTorch 2.6、torchvision 0.21、timm、transformers、scikit-learn、NumPy、Pandas、Pillow、pytest。

## 全局约束

- 论文主任务始终是恶性病例四分类：Luminal A、Luminal B、HER2+、TNBC。
- 良恶性分类只作为辅助任务；`flat5` 只作诊断对照，不能把五分类总体 Accuracy 当作分型提升证据。
- B0/B1/B2 必须保留 `metadata.csv` 中恶性 A/B/C 划分：训练 534、验证 151、测试 82。
- B1/B2 训练只能额外加入 `metadata_5class.csv` 中 191 例良性训练样本；41 例良性验证和 42 例良性测试不得进入训练。
- 文件名推导的基础病例组只用于泄漏警告，不得据此静默移动样本；论文正式实验仍需确认患者或检查级标识。
- 验证、测试和过拟合门禁必须使用确定性预处理；同一检查点重复评估应得到相同预测。
- 32 例门禁固定为每亚型 8 例，64 例门禁固定为每亚型 16 例；32 例训练 Accuracy 未达到 98% 时不得运行 64 例或新增模型创新。
- 第一轮受控比较使用加权交叉熵，不使用标签平滑、CAM、特征多样性正则、Focal Loss 或动态模态加权。
- `dual_head` 默认使用 `L_total = L_subtype + 0.3 * L_benign_malignant`，亚型损失只在恶性样本上计算。
- 模型选择主指标为 `malignant_macro_f1`，不得使用平坦五分类总体 Accuracy 选取论文主模型。
- 所有新增 Markdown、注释、文档字符串和用户可见日志使用中文；代码标识符、命令、模型名和标准指标名保留英文。
- 检查点、运行目录和大文件不得提交 Git；实现留在 `exp/reliable-stage2-baselines`，未获得用户审核前不得再次推送远端。

---

## 文件结构

### 新建文件

- `code/datasets/split_utils.py`：读取元数据、构建公平划分、选择固定过拟合子集、生成疑似病例组审计和划分清单。
- `code/datasets/bus_dataset.py`：只加载 BUS 与亚型标签，供小样本过拟合门禁使用。
- `code/models/stage2/task_heads.py`：实现 `flat4`、`flat5` 与 `dual_head` 的分类头。
- `code/models/stage2/overfit_probe.py`：实现 BUS-only TinyUSFM + Linear 的过拟合门禁模型。
- `code/train/stage2_objectives.py`：集中实现类别权重和各任务模式的损失。
- `code/train/stage2_engine.py`：实现单轮训练、确定性评估、预测语义和检查点监控。
- `code/train/run_artifacts.py`：保存 `args.json`、`split_manifest.json`、`history.json` 和 `metrics_best.json`。
- `code/tests/conftest.py`：提供无网络、临时目录的数据集与 tokenizer 测试夹具。
- `code/tests/test_split_utils.py`：验证真实数据计数、公平划分、固定子集和疑似组审计。
- `code/tests/test_stage2_objectives.py`：验证平坦与双头损失、掩码和类别权重。
- `code/tests/test_evaluation.py`：验证四分类、五分类、二分类、条件分型与端到端分型指标。
- `code/tests/test_overfit_probe.py`：验证 BUS-only 模型输出、参数可训练性和有限梯度。
- `code/tests/test_run_artifacts.py`：验证运行参数、划分清单、历史和最佳指标的保存。
- `docs/experiments/reliable-stage2-baselines.md`：记录固定命令、门禁结论和 B0/B1/B2 结果表。

### 修改文件

- `code/datasets/dataset.py`：增加确定性变换、显式 `augment`、外部样本清单和三类标签。
- `code/models/stage2/subtyping_model.py`：接入任务分类头，改为结构化字典输出。
- `code/train/train_stage2.py`：改造成统一的 `overfit|flat4|flat5|dual_head` 命令入口。
- `code/utils/evaluation.py`：去除硬编码四分类，支持配置类别、关注类别和二分类 AUC。
- `code/tests/test_dataset.py`：移除硬编码 Linux 路径与网络依赖，验证确定性和标签语义。
- `code/tests/test_stage2.py`：适配结构化输出，并正确验证“冻结层无梯度、解冻层有梯度”。
- `code/tests/test_stage1.py`：移除旧 Linux 绝对路径，缺少真实权重时使用标准跳过语义。
- `.gitignore`：忽略实验运行目录和本地划分产物，但保留人工维护的实验记录文档。

---

### 任务 1：公平数据划分与疑似患者组审计

**文件：**
- 新建：`code/datasets/split_utils.py`
- 新建：`code/tests/test_split_utils.py`

**接口：**
- 输入：`metadata.csv`、`metadata_5class.csv`，任务模式和随机种子。
- 输出：
  - `load_metadata(path: str | Path) -> list[dict]`
  - `build_fair_splits(project_root: str | Path, task_mode: str, malignant_metadata="metadata.csv", benign_metadata="metadata_5class.csv") -> dict[str, list[dict]]`
  - `derive_suspected_group_id(case_id: str) -> str`
  - `audit_cross_split_groups(samples: Sequence[Mapping]) -> list[dict]`
  - `select_balanced_subset(samples: Sequence[Mapping], per_class: int, seed: int) -> list[dict]`
  - `select_samples_by_case_ids(samples: Sequence[Mapping], case_ids: Sequence[str]) -> list[dict]`
  - `build_split_manifest(task_mode: str, splits: Mapping, seed: int) -> dict`

- [ ] **步骤 1：先写真实数据契约和合成审计用例**

在 `code/tests/test_split_utils.py` 写入：

```python
from pathlib import Path

from code.datasets.split_utils import (
    audit_cross_split_groups,
    build_fair_splits,
    build_split_manifest,
    derive_suspected_group_id,
    select_balanced_subset,
    select_samples_by_case_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _case_ids(samples):
    return {sample["case_id"] for sample in samples}


def test_fair_splits_preserve_canonical_malignant_abc():
    flat4 = build_fair_splits(PROJECT_ROOT, "flat4")
    flat5 = build_fair_splits(PROJECT_ROOT, "flat5")
    dual = build_fair_splits(PROJECT_ROOT, "dual_head")

    assert len(flat4["train"]) == 534
    assert len(flat4["malignant_val"]) == 151
    assert len(flat4["malignant_test"]) == 82
    assert len(flat5["train"]) == 534 + 191
    assert len(dual["train"]) == 534 + 191
    assert _case_ids(flat4["malignant_val"]) == _case_ids(flat5["malignant_val"])
    assert _case_ids(flat4["malignant_test"]) == _case_ids(dual["malignant_test"])
    assert sum(s["class_label"] == 4 for s in dual["train"]) == 191
    assert all(s["class_label"] != 4 for s in flat4["train"])


def test_held_out_benign_samples_never_enter_training():
    splits = build_fair_splits(PROJECT_ROOT, "dual_head")
    train_ids = _case_ids(splits["train"])
    held_out_ids = _case_ids(splits["binary_val"]) | _case_ids(splits["binary_test"])
    benign_held_out = {
        s["case_id"]
        for s in splits["binary_val"] + splits["binary_test"]
        if s["class_label"] == 4
    }
    assert len(benign_held_out) == 41 + 42
    assert train_ids.isdisjoint(benign_held_out)
    assert train_ids.isdisjoint(held_out_ids)


def test_balanced_subset_is_fixed_and_complete():
    samples = [
        {
            "case_id": f"{label}-{index}",
            "class_label": label,
            "subtype_label": label,
            "malignancy_label": 1,
            "split": "train",
            "source": "malignant",
        }
        for label in range(4)
        for index in range(20)
    ]
    first = select_balanced_subset(samples, per_class=8, seed=42)
    second = select_balanced_subset(samples, per_class=8, seed=42)
    assert first == second
    assert len(first) == 32
    assert [sum(s["subtype_label"] == c for s in first) for c in range(4)] == [8, 8, 8, 8]


def test_filename_grouping_is_warning_only():
    assert derive_suspected_group_id("1247-L") == "1247"
    assert derive_suspected_group_id("1247-R") == "1247"
    assert derive_suspected_group_id("078") == "078"
    rows = [
        {"case_id": "1247-L", "split": "train"},
        {"case_id": "1247-R", "split": "val"},
        {"case_id": "078", "split": "test"},
    ]
    audit = audit_cross_split_groups(rows)
    assert audit == [{
        "suspected_group_id": "1247",
        "case_ids": ["1247-L", "1247-R"],
        "splits": ["train", "val"],
    }]


def test_real_metadata_audit_counts_are_explicit():
    from code.datasets.split_utils import load_metadata

    data_dir = PROJECT_ROOT / "data"
    malignant = load_metadata(data_dir / "metadata.csv", "malignant")
    five_class = load_metadata(data_dir / "metadata_5class.csv", "five_class")
    assert len(audit_cross_split_groups(malignant)) == 2
    assert len(audit_cross_split_groups(five_class)) == 31


def test_manifest_has_no_duplicate_ids_and_can_restore_order():
    splits = build_fair_splits(PROJECT_ROOT, "flat4")
    manifest = build_split_manifest("flat4", splits, seed=42)
    for case_ids in manifest["splits"].values():
        assert len(case_ids) == len(set(case_ids))
    requested = manifest["splits"]["malignant_val"][:5]
    restored = select_samples_by_case_ids(splits["malignant_val"], requested)
    assert [sample["case_id"] for sample in restored] == requested
```

- [ ] **步骤 2：运行测试并确认因模块不存在而失败**

运行：

```powershell
python -m pytest code/tests/test_split_utils.py -q
```

预期：收集阶段失败，包含 `ModuleNotFoundError: No module named 'code.datasets.split_utils'`。

- [ ] **步骤 3：实现统一样本语义和公平划分**

在 `code/datasets/split_utils.py` 写入：

```python
"""实验数据划分、固定子集和疑似病例组审计工具。"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


VALID_TASK_MODES = {"overfit", "flat4", "flat5", "dual_head"}


def _normalize_row(row: Mapping[str, str], source: str) -> dict:
    class_label = int(row["subtype_label"])
    is_malignant = class_label < 4
    return {
        "case_id": row["case_id"],
        "class_label": class_label,
        "malignancy_label": int(is_malignant),
        "subtype_label": class_label if is_malignant else -1,
        "split": row["split"],
        "source": source,
    }


def load_metadata(path: str | Path, source: str) -> list[dict]:
    """读取元数据并转换为统一任务标签。"""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [_normalize_row(row, source) for row in csv.DictReader(handle)]


def _by_split(samples: Sequence[Mapping]) -> dict[str, list[dict]]:
    result = {"train": [], "val": [], "test": []}
    for sample in samples:
        result[sample["split"]].append(dict(sample))
    return result


def build_fair_splits(
    project_root: str | Path,
    task_mode: str,
    malignant_metadata: str = "metadata.csv",
    benign_metadata: str = "metadata_5class.csv",
) -> dict[str, list[dict]]:
    """按 canonical 恶性 A/B/C 构建可公平比较的任务划分。"""
    if task_mode not in VALID_TASK_MODES:
        raise ValueError(f"未知 task_mode: {task_mode}")

    data_dir = Path(project_root) / "data"
    malignant = _by_split(load_metadata(data_dir / malignant_metadata, "malignant"))
    five_class = load_metadata(data_dir / benign_metadata, "five_class")
    benign = _by_split([sample for sample in five_class if sample["class_label"] == 4])

    train = list(malignant["train"])
    if task_mode in {"flat5", "dual_head"}:
        train.extend(benign["train"])

    return {
        "train": train,
        "malignant_val": list(malignant["val"]),
        "malignant_test": list(malignant["test"]),
        "binary_val": list(malignant["val"]) + list(benign["val"]),
        "binary_test": list(malignant["test"]) + list(benign["test"]),
    }


def derive_suspected_group_id(case_id: str) -> str:
    """按首个连字符前缀推导疑似基础病例组，仅用于生成警告。"""
    return case_id.split("-", maxsplit=1)[0]


def audit_cross_split_groups(samples: Sequence[Mapping]) -> list[dict]:
    """列出跨 split 的疑似基础病例组，不修改任何样本归属。"""
    groups = defaultdict(list)
    for sample in samples:
        groups[derive_suspected_group_id(sample["case_id"])].append(sample)

    conflicts = []
    for group_id, group_samples in groups.items():
        splits = sorted({sample["split"] for sample in group_samples})
        if len(splits) > 1:
            conflicts.append({
                "suspected_group_id": group_id,
                "case_ids": sorted(sample["case_id"] for sample in group_samples),
                "splits": splits,
            })
    return sorted(conflicts, key=lambda item: item["suspected_group_id"])


def select_balanced_subset(
    samples: Sequence[Mapping],
    per_class: int,
    seed: int,
) -> list[dict]:
    """从恶性训练集为四个亚型分别固定抽取相同数量样本。"""
    rng = random.Random(seed)
    selected = []
    for class_index in range(4):
        candidates = sorted(
            (dict(s) for s in samples if s["subtype_label"] == class_index),
            key=lambda sample: sample["case_id"],
        )
        if len(candidates) < per_class:
            raise ValueError(f"亚型 {class_index} 仅有 {len(candidates)} 例，少于 {per_class}")
        selected.extend(rng.sample(candidates, per_class))
    return sorted(selected, key=lambda sample: (sample["subtype_label"], sample["case_id"]))


def select_samples_by_case_ids(
    samples: Sequence[Mapping],
    case_ids: Sequence[str],
) -> list[dict]:
    """按清单顺序恢复样本，供检查点复评和固定子集恢复使用。"""
    index = {sample["case_id"]: dict(sample) for sample in samples}
    missing = [case_id for case_id in case_ids if case_id not in index]
    if missing:
        raise ValueError(f"划分清单包含未知 case_id: {missing[:5]}")
    return [index[case_id] for case_id in case_ids]


def build_split_manifest(task_mode: str, splits: Mapping, seed: int) -> dict:
    """生成不含图像或文本内容的轻量划分清单。"""
    all_samples = [sample for values in splits.values() for sample in values]
    return {
        "task_mode": task_mode,
        "seed": seed,
        "splits": {
            name: [sample["case_id"] for sample in values]
            for name, values in splits.items()
        },
        "suspected_cross_split_groups": audit_cross_split_groups(all_samples),
    }
```

- [ ] **步骤 4：运行划分测试并核对真实计数**

运行：

```powershell
python -m pytest code/tests/test_split_utils.py -q
```

预期：`6 passed`；若疑似组数量与设计文档记录不一致，只更新审计事实，不移动样本。

- [ ] **步骤 5：提交划分工具**

```powershell
git add code/datasets/split_utils.py code/tests/test_split_utils.py
git commit -m "feat: add fair stage2 split contract"
```

---

### 任务 2：确定性数据集与显式任务标签

**文件：**
- 新建：`code/tests/conftest.py`
- 新建：`code/datasets/bus_dataset.py`
- 修改：`code/datasets/dataset.py`
- 修改：`code/tests/test_dataset.py`

**接口：**
- 输入：任务 1 生成的 `Sequence[Mapping]`。
- 输出：`MultiModalBreastDataset(root_dir, split="train", img_size=224, max_text_len=128, tokenizer_name="emilyalsentzer/Bio_ClinicalBERT", transform_paired=None, transform_cdfi=None, metadata_file="metadata.csv", samples=None, augment=None, tokenizer=None)`；每个样本返回 `case_id`、`class_label`、`malignancy_label` 和 `subtype_label`。
- 输出：`BUSOverfitDataset(root_dir, samples, img_size=224, transform=None)`；只读取 BUS，不构造 tokenizer 或访问其他模态。

- [ ] **步骤 1：创建无网络临时数据夹具**

在 `code/tests/conftest.py` 写入：

```python
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
```

- [ ] **步骤 2：用新接口重写数据集测试并确认失败**

将 `code/tests/test_dataset.py` 改为：

```python
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
```

运行：

```powershell
python -m pytest code/tests/test_dataset.py -q
```

预期：因 `PairedEvaluationTransform`、`CDFIEvaluationTransform` 或新构造参数不存在而失败。

- [ ] **步骤 3：实现确定性变换和兼容的数据集构造**

在 `code/datasets/dataset.py` 中新增：

```python
class PairedEvaluationTransform:
    """BUS 与 SWE 的确定性等比例缩放和中心裁剪。"""

    def __init__(self, img_size: int = 224):
        self.img_size = img_size

    def __call__(self, bus_img: Image.Image, swe_img: Image.Image):
        bus_img = TF.resize(bus_img, self.img_size)
        swe_img = TF.resize(swe_img, self.img_size)
        bus_img = TF.center_crop(bus_img, [self.img_size, self.img_size])
        swe_img = TF.center_crop(swe_img, [self.img_size, self.img_size])
        return bus_img, swe_img


class CDFIEvaluationTransform:
    """CDFI 的确定性缩放。"""

    def __init__(self, img_size: int = 224):
        self.transform = transforms.Resize((img_size, img_size))

    def __call__(self, cdfi_img: Image.Image):
        return self.transform(cdfi_img)
```

将构造函数扩展为：

```python
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
    samples: Optional[List[Dict]] = None,
    augment: Optional[bool] = None,
    tokenizer=None,
):
    self.root_dir = root_dir
    self.split = split
    self.img_size = img_size
    self.max_text_len = max_text_len
    self.data_dir = os.path.join(root_dir, "data")
    self.metadata_path = os.path.join(self.data_dir, metadata_file)
    self.bus_dir = os.path.join(self.data_dir, "images", "BUS")
    self.swe_dir = os.path.join(self.data_dir, "images", "SWE")
    self.cdfi_dir = os.path.join(self.data_dir, "images", "CDFI")
    self.texts_dir = os.path.join(self.data_dir, "texts")
    self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)
    self.samples = [dict(sample) for sample in samples] if samples is not None else self._load_metadata()
    self.augment = (split == "train") if augment is None else augment
    if transform_paired is None:
        transform_paired = (
            PairedAlignedTransform(img_size)
            if self.augment
            else PairedEvaluationTransform(img_size)
        )
    if transform_cdfi is None:
        transform_cdfi = (
            CDFIIndependentTransform(img_size)
            if self.augment
            else CDFIEvaluationTransform(img_size)
        )
    self.transform_paired = transform_paired
    self.transform_cdfi = transform_cdfi
```

将 `_load_metadata()` 的样本字典改成与任务 1 相同的标签语义，并在 `__getitem__()` 返回值中加入：

```python
"case_id": case_id,
"class_label": sample.get("class_label", sample["subtype_label"]),
"malignancy_label": sample.get(
    "malignancy_label",
    int(sample["subtype_label"] < 4),
),
"subtype_label": sample["subtype_label"],
```

- [ ] **步骤 4：实现 BUS-only 确定性门禁数据集**

在 `code/datasets/bus_dataset.py` 写入：

```python
"""BUS-only 小样本过拟合门禁数据集。"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BUSOverfitDataset(Dataset):
    """只加载 BUS 和恶性四分类标签，避免无关模态及 tokenizer 干扰门禁。"""

    def __init__(self, root_dir, samples, img_size=224, transform=None):
        self.bus_dir = Path(root_dir) / "data" / "images" / "BUS"
        self.samples = [dict(sample) for sample in samples]
        self.transform = transform or transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(
            self.bus_dir / f"{sample['case_id']}.jpg"
        ).convert("RGB")
        return {
            "case_id": sample["case_id"],
            "bus_img": self.transform(image),
            "subtype_label": sample["subtype_label"],
        }
```

- [ ] **步骤 5：运行数据集测试**

```powershell
python -m pytest code/tests/test_dataset.py -q
```

预期：`5 passed`，测试过程中不访问 HuggingFace 网络，BUS-only 数据集不读取 SWE、CDFI 或文本。

- [ ] **步骤 6：提交确定性数据管线**

```powershell
git add code/datasets/bus_dataset.py code/datasets/dataset.py code/tests/conftest.py code/tests/test_dataset.py
git commit -m "fix: make stage2 evaluation deterministic"
```

---

### 任务 3：通用分类评估与端到端预测语义

**文件：**
- 修改：`code/utils/evaluation.py`
- 新建：`code/tests/test_evaluation.py`

**接口：**
- `evaluate_predictions(y_true, y_pred, class_names, focus_labels=None, y_score=None, save_dir=None, figure_name="confusion_matrix.png") -> dict`
- `conditional_malignant_predictions(class_logits: Tensor) -> ndarray`
- `end_to_end_flat5_predictions(class_logits: Tensor) -> ndarray`
- `end_to_end_dual_predictions(malignancy_logits: Tensor, subtype_logits: Tensor) -> ndarray`

- [ ] **步骤 1：写四种评估语义的失败测试**

在 `code/tests/test_evaluation.py` 写入：

```python
import numpy as np
import torch

from code.utils.evaluation import (
    conditional_malignant_predictions,
    end_to_end_dual_predictions,
    end_to_end_flat5_predictions,
    evaluate_predictions,
)


SUBTYPES = ["Luminal A", "Luminal B", "HER2+", "TNBC"]


def test_four_class_metrics_have_configured_dimensions():
    result = evaluate_predictions(
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 1, 3]),
        class_names=SUBTYPES,
    )
    assert result["accuracy"] == 0.75
    assert len(result["confusion_matrix"]) == 4
    assert set(result["per_class"]) == set(SUBTYPES)
    assert "balanced_accuracy" in result
    assert result["prediction_distribution"] == [1, 2, 0, 1]


def test_flat5_conditional_and_end_to_end_predictions_differ():
    logits = torch.tensor([
        [4.0, 1.0, 0.0, 0.0, 5.0],
        [0.0, 4.0, 1.0, 0.0, 2.0],
    ])
    assert conditional_malignant_predictions(logits).tolist() == [0, 1]
    assert end_to_end_flat5_predictions(logits).tolist() == [4, 1]


def test_dual_head_end_to_end_marks_binary_benign_as_class_four():
    malignancy_logits = torch.tensor([[5.0, 1.0], [1.0, 5.0]])
    subtype_logits = torch.tensor([[0.0, 0.0, 5.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
    assert end_to_end_dual_predictions(
        malignancy_logits, subtype_logits
    ).tolist() == [4, 1]


def test_binary_auc_is_reported():
    result = evaluate_predictions(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        class_names=["Benign", "Malignant"],
        y_score=np.array([0.1, 0.6, 0.8, 0.9]),
    )
    assert result["auc"] == 1.0
    assert result["per_class"]["Malignant"]["recall"] == 1.0
```

- [ ] **步骤 2：运行测试并确认旧评估接口失败**

```powershell
python -m pytest code/tests/test_evaluation.py -q
```

预期：失败信息包含未知参数 `class_names` 或缺失预测转换函数。

- [ ] **步骤 3：将评估实现改成配置驱动**

在 `code/utils/evaluation.py` 中保留绘图能力，核心实现改为：

```python
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true,
    y_pred,
    class_names,
    focus_labels=None,
    y_score=None,
    save_dir=None,
    figure_name="confusion_matrix.png",
):
    """按配置类别计算分类指标，可将宏平均限制在关注类别。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    focus_labels = labels if focus_labels is None else list(focus_labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for index, name in enumerate(class_names):
        true_positive = int(((y_true == index) & (y_pred == index)).sum())
        false_negative = int(((y_true == index) & (y_pred != index)).sum())
        false_positive = int(((y_true != index) & (y_pred == index)).sum())
        true_negative = int(((y_true != index) & (y_pred != index)).sum())
        per_class[name] = {
            "recall": true_positive / max(true_positive + false_negative, 1),
            "precision": true_positive / max(true_positive + false_positive, 1),
            "specificity": true_negative / max(true_negative + false_positive, 1),
            "f1": (
                2 * true_positive
                / max(2 * true_positive + false_positive + false_negative, 1)
            ),
        }

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            recall_score(
                y_true,
                y_pred,
                labels=focus_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, labels=focus_labels, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=focus_labels, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, labels=focus_labels, average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred, labels=focus_labels, average="weighted", zero_division=0
            )
        ),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "label_distribution": np.bincount(
            y_true, minlength=len(class_names)
        ).tolist(),
        "prediction_distribution": np.bincount(
            y_pred, minlength=len(class_names)
        ).tolist(),
    }
    if y_score is not None:
        result["auc"] = float(roc_auc_score(y_true, np.asarray(y_score)))
    if save_dir:
        _plot_confusion_matrix(cm, class_names, save_dir, figure_name)
    return result


def conditional_malignant_predictions(class_logits):
    """忽略良性 logit，仅在四个恶性亚型中决策。"""
    return class_logits[:, :4].argmax(dim=1).detach().cpu().numpy()


def end_to_end_flat5_predictions(class_logits):
    """平坦五分类端到端预测，类别 4 表示预测为良性。"""
    return class_logits.argmax(dim=1).detach().cpu().numpy()


def end_to_end_dual_predictions(malignancy_logits, subtype_logits):
    """双头端到端预测：二分类判为良性时输出类别 4。"""
    binary = malignancy_logits.argmax(dim=1)
    subtype = subtype_logits.argmax(dim=1)
    combined = torch.where(binary == 0, torch.full_like(subtype, 4), subtype)
    return combined.detach().cpu().numpy()
```

同步修改 `_plot_confusion_matrix`，显式接收 `class_names` 和 `figure_name`，不再读取全局 `SUBTYPE_NAMES`。

- [ ] **步骤 4：运行评估测试**

```powershell
python -m pytest code/tests/test_evaluation.py -q
```

预期：`4 passed`。

- [ ] **步骤 5：提交通用评估**

```powershell
git add code/utils/evaluation.py code/tests/test_evaluation.py
git commit -m "feat: add task-aware stage2 evaluation"
```

---

### 任务 4：BUS-only TinyUSFM 过拟合门禁模型

**文件：**
- 新建：`code/models/stage2/overfit_probe.py`
- 新建：`code/tests/test_overfit_probe.py`

**接口：**
- `BUSOverfitProbe(pretrained_path=None, num_classes=4, img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=12)`
- `BUSOverfitProbe.forward(bus_img: Tensor) -> dict[str, Tensor]`

- [ ] **步骤 1：先写输出和梯度门禁测试**

在 `code/tests/test_overfit_probe.py` 写入：

```python
import torch

from code.models.stage2.overfit_probe import BUSOverfitProbe


def test_overfit_probe_returns_four_class_logits():
    model = BUSOverfitProbe(
        pretrained_path=None,
        img_size=32,
        patch_size=16,
        embed_dim=48,
        depth=2,
        num_heads=4,
    )
    output = model(torch.randn(4, 1, 32, 32))
    assert output["class_logits"].shape == (4, 4)
    assert output["image_features"].shape == (4, 48)


def test_all_overfit_probe_parameters_receive_finite_gradients():
    model = BUSOverfitProbe(
        pretrained_path=None,
        img_size=32,
        patch_size=16,
        embed_dim=48,
        depth=2,
        num_heads=4,
    )
    labels = torch.tensor([0, 1, 2, 3])
    loss = torch.nn.functional.cross_entropy(
        model(torch.randn(4, 1, 32, 32))["class_logits"],
        labels,
    )
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert all(parameter.grad.abs().sum() > 0 for parameter in trainable)
```

- [ ] **步骤 2：运行测试并确认模型模块不存在**

```powershell
python -m pytest code/tests/test_overfit_probe.py -q
```

预期：`ModuleNotFoundError`。

- [ ] **步骤 3：实现全量可训练的 BUS-only 模型**

在 `code/models/stage2/overfit_probe.py` 写入：

```python
"""用于验证标签、损失、梯度和优化器链路的 BUS-only 过拟合模型。"""

import os

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class BUSOverfitProbe(nn.Module):
    """单通道 TinyUSFM 编码器与线性四分类头，全部参数可训练。"""

    def __init__(
        self,
        pretrained_path=None,
        num_classes=4,
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=12,
    ):
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0,
            qkv_bias=True,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        encoder_state = self.encoder.state_dict()
        compatible = {}
        for key, value in state_dict.items():
            name = (
                key.replace("model.", "")
                .replace("module.", "")
                .replace("backbone.", "")
                .replace("encoder.", "")
            )
            if "patch_embed" in name:
                continue
            if name in encoder_state and encoder_state[name].shape == value.shape:
                compatible[name] = value
        self.encoder.load_state_dict(compatible, strict=False)

    def forward(self, bus_img):
        features = self.encoder.forward_features(bus_img)[:, 0]
        return {
            "class_logits": self.classifier(features),
            "image_features": features,
        }
```

- [ ] **步骤 4：运行模型测试**

```powershell
python -m pytest code/tests/test_overfit_probe.py -q
```

预期：`2 passed`，所有可训练参数梯度非零且有限。

- [ ] **步骤 5：提交门禁模型**

```powershell
git add code/models/stage2/overfit_probe.py code/tests/test_overfit_probe.py
git commit -m "feat: add bus-only overfit probe"
```

---

### 任务 5：任务分类头与双头损失

**文件：**
- 新建：`code/models/stage2/task_heads.py`
- 新建：`code/train/stage2_objectives.py`
- 新建：`code/tests/test_stage2_objectives.py`

**接口：**
- `Stage2TaskHeads(input_dim: int, task_mode: str, hidden_dim: int = 128)`
- `compute_class_weights(samples, label_key, num_classes, ignore_index=None) -> Tensor`
- `compute_stage2_losses(outputs, batch, task_mode, criteria, lambda_bm=0.3) -> dict[str, Tensor]`

- [ ] **步骤 1：写任务头和掩码损失测试**

在 `code/tests/test_stage2_objectives.py` 写入：

```python
import torch
import torch.nn as nn

from code.models.stage2.task_heads import Stage2TaskHeads
from code.train.stage2_objectives import (
    compute_class_weights,
    compute_stage2_losses,
)


def test_task_heads_expose_expected_logits():
    features = torch.randn(3, 16)
    assert Stage2TaskHeads(16, "flat4")(features)["class_logits"].shape == (3, 4)
    assert Stage2TaskHeads(16, "flat5")(features)["class_logits"].shape == (3, 5)
    dual = Stage2TaskHeads(16, "dual_head")(features)
    assert dual["malignancy_logits"].shape == (3, 2)
    assert dual["subtype_logits"].shape == (3, 4)


def test_dual_head_subtype_loss_ignores_benign():
    outputs = {
        "malignancy_logits": torch.tensor(
            [[3.0, 0.0], [0.0, 3.0], [0.0, 3.0]], requires_grad=True
        ),
        "subtype_logits": torch.tensor(
            [[9.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]],
            requires_grad=True,
        ),
    }
    batch = {
        "malignancy_label": torch.tensor([0, 1, 1]),
        "subtype_label": torch.tensor([-1, 0, 1]),
    }
    criteria = {
        "malignancy": nn.CrossEntropyLoss(),
        "subtype": nn.CrossEntropyLoss(),
    }
    losses = compute_stage2_losses(
        outputs, batch, "dual_head", criteria, lambda_bm=0.3
    )
    expected_subtype = criteria["subtype"](
        outputs["subtype_logits"][1:], torch.tensor([0, 1])
    )
    assert torch.allclose(losses["subtype_loss"], expected_subtype)
    assert torch.allclose(
        losses["total_loss"],
        losses["subtype_loss"] + 0.3 * losses["malignancy_loss"],
    )


def test_benign_only_batch_has_finite_zero_subtype_loss():
    outputs = {
        "malignancy_logits": torch.randn(2, 2, requires_grad=True),
        "subtype_logits": torch.randn(2, 4, requires_grad=True),
    }
    batch = {
        "malignancy_label": torch.zeros(2, dtype=torch.long),
        "subtype_label": torch.full((2,), -1, dtype=torch.long),
    }
    criteria = {
        "malignancy": nn.CrossEntropyLoss(),
        "subtype": nn.CrossEntropyLoss(),
    }
    losses = compute_stage2_losses(outputs, batch, "dual_head", criteria)
    assert losses["subtype_loss"].item() == 0.0
    assert torch.isfinite(losses["total_loss"])
    losses["total_loss"].backward()
    assert outputs["subtype_logits"].grad is not None


def test_separate_class_weights_have_expected_shapes():
    samples = [
        {"malignancy_label": 0, "subtype_label": -1},
        {"malignancy_label": 1, "subtype_label": 0},
        {"malignancy_label": 1, "subtype_label": 1},
        {"malignancy_label": 1, "subtype_label": 1},
    ]
    binary = compute_class_weights(samples, "malignancy_label", 2)
    subtype = compute_class_weights(samples, "subtype_label", 4, ignore_index=-1)
    assert binary.shape == (2,)
    assert subtype.shape == (4,)
    assert torch.isfinite(binary).all()
    assert torch.isfinite(subtype).all()
```

- [ ] **步骤 2：运行测试并确认模块缺失**

```powershell
python -m pytest code/tests/test_stage2_objectives.py -q
```

预期：收集阶段失败。

- [ ] **步骤 3：实现配置驱动的任务头**

在 `code/models/stage2/task_heads.py` 写入：

```python
"""Stage 2 平坦分类头与层级双任务头。"""

import torch.nn as nn


def _mlp(input_dim, output_dim, hidden_dim):
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, output_dim),
    )


class Stage2TaskHeads(nn.Module):
    """根据任务模式输出平坦 logits 或良恶性与亚型双头 logits。"""

    def __init__(self, input_dim, task_mode, hidden_dim=128):
        super().__init__()
        if task_mode not in {"flat4", "flat5", "dual_head"}:
            raise ValueError(f"任务分类头不支持模式: {task_mode}")
        self.task_mode = task_mode
        if task_mode == "dual_head":
            self.malignancy_head = _mlp(input_dim, 2, hidden_dim)
            self.subtype_head = _mlp(input_dim, 4, hidden_dim)
        else:
            self.class_head = _mlp(
                input_dim,
                4 if task_mode == "flat4" else 5,
                hidden_dim,
            )

    def forward(self, features):
        if self.task_mode == "dual_head":
            return {
                "malignancy_logits": self.malignancy_head(features),
                "subtype_logits": self.subtype_head(features),
            }
        return {"class_logits": self.class_head(features)}
```

- [ ] **步骤 4：实现类别权重和双头目标**

在 `code/train/stage2_objectives.py` 写入：

```python
"""Stage 2 类别权重与任务损失。"""

from collections import Counter

import torch


def compute_class_weights(
    samples,
    label_key,
    num_classes,
    ignore_index=None,
    device="cpu",
):
    labels = [
        int(sample[label_key])
        for sample in samples
        if ignore_index is None or int(sample[label_key]) != ignore_index
    ]
    counts = Counter(labels)
    weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    total = len(labels)
    for class_index in range(num_classes):
        count = counts.get(class_index, 0)
        if count > 0:
            weights[class_index] = total / (num_classes * count)
    return weights


def compute_stage2_losses(
    outputs,
    batch,
    task_mode,
    criteria,
    lambda_bm=0.3,
):
    if task_mode == "flat4":
        loss = criteria["class"](outputs["class_logits"], batch["subtype_label"])
        return {"total_loss": loss, "class_loss": loss}

    if task_mode == "flat5":
        loss = criteria["class"](outputs["class_logits"], batch["class_label"])
        return {"total_loss": loss, "class_loss": loss}

    if task_mode != "dual_head":
        raise ValueError(f"未知 task_mode: {task_mode}")

    malignancy_loss = criteria["malignancy"](
        outputs["malignancy_logits"],
        batch["malignancy_label"],
    )
    malignant_mask = batch["malignancy_label"] == 1
    if malignant_mask.any():
        subtype_loss = criteria["subtype"](
            outputs["subtype_logits"][malignant_mask],
            batch["subtype_label"][malignant_mask],
        )
    else:
        subtype_loss = outputs["subtype_logits"].sum() * 0.0
    total_loss = subtype_loss + lambda_bm * malignancy_loss
    return {
        "total_loss": total_loss,
        "subtype_loss": subtype_loss,
        "malignancy_loss": malignancy_loss,
    }
```

- [ ] **步骤 5：运行损失测试**

```powershell
python -m pytest code/tests/test_stage2_objectives.py -q
```

预期：`4 passed`。

- [ ] **步骤 6：提交分类头和损失**

```powershell
git add code/models/stage2/task_heads.py code/train/stage2_objectives.py code/tests/test_stage2_objectives.py
git commit -m "feat: add dual-head stage2 objective"
```

---

### 任务 6：Stage 2 模型结构化输出与正确梯度检查

**文件：**
- 修改：`code/models/stage2/subtyping_model.py`
- 修改：`code/tests/test_stage2.py`
- 修改：`code/tests/test_stage1.py`

**接口：**
- `SECSubtypingModel(pretrained_path=None, task_mode="flat4", embed_dim=192, patch_size=16, img_size=224, depth=12, num_heads=12, ip_adapter_layers=None, ip_adapter_scale=0.1, unfreeze_last_n=2)`
- `forward(bus_img, swe_img, cdfi_img, input_ids, attention_mask) -> {"class_logits"|双头 logits, "image_features", "text_features", "fused_features"}`
- `model.task_heads.parameters()` 供优化器统一收集。

- [ ] **步骤 1：先改测试，固定结构化返回值和梯度语义**

在 `code/tests/test_stage2.py` 中保留独立模块维度测试；用 monkeypatch 避免下载 BERT，并将完整模型测试替换为：

```python
import torch
import torch.nn as nn

import code.models.stage2.subtyping_model as subtyping_module


class FakeTextBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 512)

    def forward(self, input_ids, attention_mask):
        pooled = torch.nn.functional.one_hot(
            input_ids[:, 0] % 8, num_classes=8
        ).float()
        return self.projection(pooled)


def _inputs(batch_size=2):
    return (
        torch.randn(batch_size, 1, 32, 32),
        torch.randn(batch_size, 3, 32, 32),
        torch.randn(batch_size, 3, 32, 32),
        torch.randint(0, 100, (batch_size, 8)),
        torch.ones(batch_size, 8, dtype=torch.long),
    )


def test_full_model_returns_task_aware_dictionary(monkeypatch):
    monkeypatch.setattr(subtyping_module, "TextLogicBranch", FakeTextBranch)
    for task_mode in ("flat4", "flat5", "dual_head"):
        model = subtyping_module.SECSubtypingModel(
            pretrained_path=None,
            task_mode=task_mode,
            img_size=32,
            patch_size=16,
            embed_dim=48,
            depth=2,
            num_heads=4,
            ip_adapter_layers=[1],
            unfreeze_last_n=1,
        )
        output = model(*_inputs())
        assert output["image_features"].shape == (2, 48)
        assert output["text_features"].shape == (2, 512)
        assert output["fused_features"].shape == (2, 560)
        if task_mode == "flat4":
            assert output["class_logits"].shape == (2, 4)
        elif task_mode == "flat5":
            assert output["class_logits"].shape == (2, 5)
        else:
            assert output["malignancy_logits"].shape == (2, 2)
            assert output["subtype_logits"].shape == (2, 4)


def test_only_configured_encoder_layers_receive_gradients(monkeypatch):
    monkeypatch.setattr(subtyping_module, "TextLogicBranch", FakeTextBranch)
    model = subtyping_module.SECSubtypingModel(
        pretrained_path=None,
        task_mode="dual_head",
        img_size=32,
        patch_size=16,
        embed_dim=48,
        depth=2,
        num_heads=4,
        ip_adapter_layers=[1],
        unfreeze_last_n=1,
    )
    output = model(*_inputs())
    (output["malignancy_logits"].sum() + output["subtype_logits"].sum()).backward()
    assert all(parameter.grad is None for parameter in model.encoder.blocks[0].parameters())
    assert any(parameter.grad is not None for parameter in model.encoder.blocks[1].parameters())
    assert any(parameter.grad is not None for parameter in model.task_heads.parameters())
    assert any(parameter.grad is not None for parameter in model.ip_adapters.parameters())
```

- [ ] **步骤 2：移除测试中的网络依赖和旧绝对路径**

在 `code/tests/test_stage2.py` 中用以下假 BERT 测试 `TextLogicBranch`，不要实例化远端模型：

```python
from types import SimpleNamespace

import code.models.stage2.text_branch as text_branch_module


class FakeBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 768)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_text_branch_without_network(monkeypatch):
    monkeypatch.setattr(
        text_branch_module.AutoModel,
        "from_pretrained",
        lambda _name: FakeBert(),
    )
    model = text_branch_module.TextLogicBranch()
    output = model(
        torch.randint(0, 1000, (2, 8)),
        torch.ones(2, 8, dtype=torch.long),
    )
    assert output.shape == (2, 512)
    assert all(not parameter.requires_grad for parameter in model.bert.parameters())
    assert all(parameter.requires_grad for parameter in model.projection.parameters())
```

在 `code/tests/test_stage1.py` 中加入 `from pathlib import Path` 和 `import pytest`，并将预训练权重路径与缺失处理改为：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
pretrained = PROJECT_ROOT / "TinyUSFM.pth"
if not pretrained.exists():
    pytest.skip("仓库根目录未提供 TinyUSFM.pth")
model = ACMMIMPretrainModel(pretrained_path=str(pretrained))
```

- [ ] **步骤 3：运行完整模型测试并确认旧 tuple 接口失败**

```powershell
python -m pytest code/tests/test_stage2.py code/tests/test_stage1.py -q
```

预期：Stage 2 完整模型相关测试失败，提示未知 `task_mode` 或返回值不是字典；Stage 1 的真实权重测试在权重缺失时显示 `SKIPPED`。

- [ ] **步骤 4：接入任务头并返回结构化结果**

在 `code/models/stage2/subtyping_model.py`：

```python
from .task_heads import Stage2TaskHeads
```

构造函数增加 `task_mode: str = "flat4"`，删除 `mlp_head` 的创建，替换为：

```python
self.task_mode = task_mode
self.task_heads = Stage2TaskHeads(
    input_dim=embed_dim + 512,
    task_mode=task_mode,
)
```

将前传末尾替换为：

```python
task_outputs = self.task_heads(fused)
return {
    **task_outputs,
    "image_features": F_bus_swe,
    "text_features": F_text,
    "fused_features": fused,
}
```

同步把类文档中的“一个 N 分类 MLP”改为“配置驱动的平坦分类头或双任务头”；新注释全部使用中文。

- [ ] **步骤 5：运行 Stage 2 模型测试**

```powershell
python -m pytest code/tests/test_stage2.py -q
```

预期：独立支路、三种任务输出和冻结/解冻梯度测试全部通过。

- [ ] **步骤 6：运行截至当前的快速回归测试**

```powershell
python -m pytest code/tests/test_dataset.py code/tests/test_split_utils.py code/tests/test_evaluation.py code/tests/test_stage2_objectives.py code/tests/test_overfit_probe.py code/tests/test_stage2.py -q
```

预期：全部通过，无网络下载。

- [ ] **步骤 7：提交结构化模型接口**

```powershell
git add code/models/stage2/subtyping_model.py code/tests/test_stage2.py code/tests/test_stage1.py
git commit -m "refactor: return structured stage2 outputs"
```

---

### 任务 7：训练引擎、模型选择与运行产物

**文件：**
- 新建：`code/train/stage2_engine.py`
- 新建：`code/train/run_artifacts.py`
- 新建：`code/tests/test_run_artifacts.py`

**接口：**
- `move_batch_to_device(batch, device) -> dict`
- `train_one_epoch(model, loader, optimizer, criteria, task_mode, device, lambda_bm, max_grad_norm, accum_steps) -> dict`
- `evaluate_loader(model, loader, task_mode, device, split_kind) -> dict`
- `monitor_value(metrics, monitor_metric) -> float`
- `save_json(path, payload) -> None`
- `initialize_run_artifacts(output_dir, args, manifest) -> None`
- `save_training_state(output_dir, history, best_metrics) -> None`

- [ ] **步骤 1：先写产物与监控指标测试**

在 `code/tests/test_run_artifacts.py` 写入：

```python
import json
from argparse import Namespace

from code.train.run_artifacts import (
    initialize_run_artifacts,
    save_training_state,
)
from code.train.stage2_engine import monitor_value


def test_run_artifacts_write_required_json(tmp_path):
    args = Namespace(task_mode="dual_head", seed=42)
    manifest = {"task_mode": "dual_head", "splits": {"train": ["001"]}}
    initialize_run_artifacts(tmp_path, args, manifest)
    save_training_state(
        tmp_path,
        history=[{"epoch": 1, "train": {"total_loss": 1.0}}],
        best_metrics={"malignant_macro_f1": 0.4},
    )
    assert json.loads((tmp_path / "args.json").read_text(encoding="utf-8"))["seed"] == 42
    assert json.loads(
        (tmp_path / "split_manifest.json").read_text(encoding="utf-8")
    ) == manifest
    assert (tmp_path / "history.json").exists()
    assert (tmp_path / "metrics_best.json").exists()


def test_monitor_uses_malignant_macro_f1():
    metrics = {
        "malignant": {"macro_f1": 0.41, "balanced_accuracy": 0.43},
        "overall": {"accuracy": 0.72},
    }
    assert monitor_value(metrics, "malignant_macro_f1") == 0.41
    assert monitor_value(metrics, "balanced_acc") == 0.43
    assert monitor_value(metrics, "acc") == 0.72
```

- [ ] **步骤 2：运行测试并确认模块不存在**

```powershell
python -m pytest code/tests/test_run_artifacts.py -q
```

预期：收集阶段失败。

- [ ] **步骤 3：实现轻量 JSON 产物**

在 `code/train/run_artifacts.py` 写入：

```python
"""训练运行的轻量 JSON 产物管理。"""

import json
from pathlib import Path


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"无法序列化类型: {type(value).__name__}")


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def initialize_run_artifacts(output_dir, args, manifest):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "args.json", vars(args))
    save_json(output_dir / "split_manifest.json", manifest)


def save_training_state(output_dir, history, best_metrics):
    output_dir = Path(output_dir)
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "metrics_best.json", best_metrics)
```

- [ ] **步骤 4：实现任务感知训练与评估引擎**

在 `code/train/stage2_engine.py` 写入以下完整职责：

```python
"""Stage 2 单轮训练、评估和模型选择工具。"""

import numpy as np
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


def move_batch_to_device(batch, device):
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
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {}
    sample_count = 0
    correct = 0
    for step, raw_batch in enumerate(tqdm(loader, desc="训练", leave=False)):
        batch = move_batch_to_device(raw_batch, device)
        if task_mode == "overfit":
            outputs = model(batch["bus_img"])
            losses = {
                "total_loss": criteria["class"](
                    outputs["class_logits"], batch["subtype_label"]
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
            if task_mode == "flat5":
                predictions = outputs["class_logits"].argmax(dim=1)
                labels = batch["class_label"]
            elif task_mode == "flat4":
                predictions = outputs["class_logits"].argmax(dim=1)
                labels = batch["subtype_label"]
            else:
                malignant = batch["malignancy_label"] == 1
                predictions = outputs["subtype_logits"][malignant].argmax(dim=1)
                labels = batch["subtype_label"][malignant]

        (losses["total_loss"] / accum_steps).backward()
        if (step + 1) % accum_steps == 0 or step + 1 == len(loader):
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        correct += int((predictions == labels).sum())
        sample_count += int(labels.numel())

    return {
        **{name: value / len(loader) for name, value in totals.items()},
        "accuracy": correct / max(sample_count, 1),
    }


def _cat(values):
    return torch.cat(values, dim=0)


@torch.no_grad()
def evaluate_loader(model, loader, task_mode, device, split_kind):
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
        for key in labels:
            label_key = f"{key}_label"
            logit_key = f"{key}_logits"
            if label_key in batch:
                labels[key].append(batch[label_key].cpu())
            if logit_key in outputs:
                logits[key].append(outputs[logit_key].cpu())
        if "image_features" in outputs:
            image_features.append(outputs["image_features"].cpu())

    diagnostic = {}
    merged_logits = [
        torch.cat(values, dim=0)
        for values in logits.values()
        if values
    ]
    if merged_logits:
        merged = torch.cat(merged_logits, dim=1)
        diagnostic["logits_var"] = float(merged.var(dim=0).mean())
    if image_features:
        diagnostic["feature_var"] = float(_cat(image_features).var(dim=0).mean())

    if task_mode in {"overfit", "flat4"}:
        class_logits = _cat(logits["class"])
        y_true = _cat(labels["subtype"]).numpy()
        malignant = evaluate_predictions(
            y_true,
            class_logits.argmax(dim=1).numpy(),
            SUBTYPE_NAMES,
        )
        return {"malignant": malignant, "diagnostic": diagnostic}

    if task_mode == "flat5":
        class_logits = _cat(logits["class"])
        if split_kind == "malignant":
            y_true = _cat(labels["subtype"]).numpy()
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
        y_true = _cat(labels["malignancy"]).numpy()
        probabilities = class_logits.softmax(dim=1)
        binary_predictions = (class_logits.argmax(dim=1) != 4).long().numpy()
        binary = evaluate_predictions(
            y_true,
            binary_predictions,
            BINARY_NAMES,
            y_score=probabilities[:, :4].sum(dim=1).numpy(),
        )
        return {"binary": binary, "diagnostic": diagnostic}

    malignancy_logits = _cat(logits["malignancy"])
    subtype_logits = _cat(logits["subtype"])
    if split_kind == "malignant":
        y_true = _cat(labels["subtype"]).numpy()
        conditional = evaluate_predictions(
            y_true,
            subtype_logits.argmax(dim=1).numpy(),
            SUBTYPE_NAMES,
        )
        end_to_end = evaluate_predictions(
            y_true,
            end_to_end_dual_predictions(malignancy_logits, subtype_logits),
            FIVE_CLASS_NAMES,
            focus_labels=[0, 1, 2, 3],
        )
        return {
            "malignant": conditional,
            "malignant_end_to_end": end_to_end,
            "diagnostic": diagnostic,
        }

    y_true = _cat(labels["malignancy"]).numpy()
    probabilities = malignancy_logits.softmax(dim=1)[:, 1].numpy()
    binary = evaluate_predictions(
        y_true,
        malignancy_logits.argmax(dim=1).numpy(),
        BINARY_NAMES,
        y_score=probabilities,
    )
    return {"binary": binary, "diagnostic": diagnostic}


def monitor_value(metrics, monitor_metric):
    if monitor_metric == "malignant_macro_f1":
        return float(metrics["malignant"]["macro_f1"])
    if monitor_metric == "macro_f1":
        section = metrics.get("malignant", metrics.get("overall", metrics.get("binary")))
        return float(section["macro_f1"])
    if monitor_metric == "balanced_acc":
        section = metrics.get("malignant", metrics.get("overall", metrics.get("binary")))
        return float(section["balanced_accuracy"])
    if monitor_metric == "acc":
        section = metrics.get("overall", metrics.get("malignant", metrics.get("binary")))
        return float(section["accuracy"])
    raise ValueError(f"未知 monitor_metric: {monitor_metric}")
```

- [ ] **步骤 5：运行产物与监控测试**

```powershell
python -m pytest code/tests/test_run_artifacts.py -q
```

预期：`2 passed`。

- [ ] **步骤 6：提交训练引擎**

```powershell
git add code/train/stage2_engine.py code/train/run_artifacts.py code/tests/test_run_artifacts.py
git commit -m "feat: add reproducible stage2 training engine"
```

---

### 任务 8：统一命令入口和过拟合门禁执行

**文件：**
- 修改：`code/train/train_stage2.py`
- 修改：`.gitignore`
- 新建：`docs/experiments/reliable-stage2-baselines.md`

**接口：**
- CLI 必须提供：
  - `--task_mode overfit|flat4|flat5|dual_head`
  - `--malignant_metadata metadata.csv`
  - `--benign_metadata metadata_5class.csv`
  - `--augment` / `--no_augment`
  - `--overfit_samples 32|64`
  - `--lambda_bm 0.3`
  - `--sampler none|balanced`
  - `--monitor_metric malignant_macro_f1|macro_f1|balanced_acc|acc`
  - `--eval_malignant_subset`

- [ ] **步骤 1：先写 CLI 解析和过拟合配置测试**

在 `code/tests/test_run_artifacts.py` 追加：

```python
from code.train.train_stage2 import build_parser


def test_cli_defaults_to_reliable_flat4_configuration():
    parser = build_parser()
    args = parser.parse_args(["--pretrained_path", "TinyUSFM.pth"])
    assert args.task_mode == "flat4"
    assert args.lambda_bm == 0.3
    assert args.monitor_metric == "malignant_macro_f1"
    assert args.beta == 0.0
    assert args.gamma == 0.0
    assert args.label_smoothing == 0.0


def test_overfit_cli_disables_regularization_and_augmentation():
    parser = build_parser()
    args = parser.parse_args([
        "--pretrained_path",
        "TinyUSFM.pth",
        "--task_mode",
        "overfit",
        "--overfit_samples",
        "32",
        "--no_augment",
    ])
    assert args.overfit_samples == 32
    assert args.augment is False
```

- [ ] **步骤 2：运行 CLI 测试并确认 `build_parser` 不存在**

```powershell
python -m pytest code/tests/test_run_artifacts.py -q
```

预期：导入失败或提示 `build_parser` 不存在。

- [ ] **步骤 3：重构 `train_stage2.py` 的参数和数据入口**

将参数创建抽成 `build_parser()`，至少包含：

```python
def build_parser():
    parser = argparse.ArgumentParser(description="Stage 2 可靠基线训练")
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument(
        "--task_mode",
        choices=["overfit", "flat4", "flat5", "dual_head"],
        default="flat4",
    )
    parser.add_argument("--malignant_metadata", default="metadata.csv")
    parser.add_argument("--benign_metadata", default="metadata_5class.csv")
    parser.add_argument("--overfit_samples", type=int, choices=[32, 64], default=32)
    parser.add_argument("--lambda_bm", type=float, default=0.3)
    parser.add_argument("--sampler", choices=["none", "balanced"], default="none")
    parser.add_argument(
        "--monitor_metric",
        choices=["malignant_macro_f1", "macro_f1", "balanced_acc", "acc"],
        default="malignant_macro_f1",
    )
    parser.add_argument(
        "--eval_malignant_subset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--evaluate_checkpoint")
    parser.add_argument("--eval_output", default="metrics_eval.json")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--unfreeze", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", default="runs/stage2")
    return parser
```

同步更新导入：

```python
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler

from code.datasets.bus_dataset import BUSOverfitDataset
from code.datasets.split_utils import (
    build_fair_splits,
    build_split_manifest,
    select_balanced_subset,
    select_samples_by_case_ids,
)
from code.models.stage2.overfit_probe import BUSOverfitProbe
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
```

`main(args)` 必须按以下顺序组装：

```python
seed_everything(args.seed)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
splits = build_fair_splits(
    PROJECT_ROOT,
    args.task_mode,
    malignant_metadata=args.malignant_metadata,
    benign_metadata=args.benign_metadata,
)

if args.task_mode == "overfit":
    per_class = args.overfit_samples // 4
    splits = {
        "train": select_balanced_subset(
            splits["train"],
            per_class=per_class,
            seed=args.seed,
        )
    }
    augment = False
else:
    augment = args.augment

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = Path(PROJECT_ROOT) / args.output_root / f"{args.task_mode}_{timestamp}"
manifest = build_split_manifest(args.task_mode, splits, args.seed)
initialize_run_artifacts(output_dir, args, manifest)
```

数据集和模型创建遵循：

```python
if args.task_mode == "overfit":
    train_dataset = BUSOverfitDataset(
        PROJECT_ROOT,
        samples=splits["train"],
        img_size=args.img_size,
    )
    model = BUSOverfitProbe(
        pretrained_path=args.pretrained_path,
        img_size=args.img_size,
    ).to(device)
    criteria = {"class": nn.CrossEntropyLoss()}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )
else:
    train_dataset = MultiModalBreastDataset(
        PROJECT_ROOT,
        split="train",
        img_size=args.img_size,
        max_text_len=args.max_text_len,
        samples=splits["train"],
        augment=augment,
    )
    model = SECSubtypingModel(
        pretrained_path=args.pretrained_path,
        task_mode=args.task_mode,
        img_size=args.img_size,
        unfreeze_last_n=args.unfreeze,
    ).to(device)
    criteria = build_criteria_for_mode(
        args.task_mode,
        train_dataset.samples,
        device,
        label_smoothing=args.label_smoothing,
    )
    optimizer = build_optimizer(model, args.lr, args.wd)
```

在同一文件中完整实现损失构造，确保二分类与亚型权重互不混用：

```python
def build_criteria_for_mode(
    task_mode,
    samples,
    device,
    label_smoothing=0.0,
):
    if task_mode == "overfit":
        return {"class": nn.CrossEntropyLoss()}
    if task_mode == "flat4":
        weights = compute_class_weights(
            samples, "subtype_label", 4, device=device
        )
        return {
            "class": nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=label_smoothing,
            )
        }
    if task_mode == "flat5":
        weights = compute_class_weights(
            samples, "class_label", 5, device=device
        )
        return {
            "class": nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=label_smoothing,
            )
        }
    return {
        "malignancy": nn.CrossEntropyLoss(
            weight=compute_class_weights(
                samples, "malignancy_label", 2, device=device
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
```

实现可选平衡采样器；`none` 使用随机打乱，`balanced` 按训练任务标签的逆频率采样：

```python
def build_train_loader(dataset, task_mode, args):
    sampler = None
    if args.sampler == "balanced":
        if task_mode == "overfit":
            raise ValueError("过拟合门禁已经分层均衡，不允许重复使用 balanced sampler")
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
```

`build_optimizer` 改为从 `model.task_heads`、`ip_adapters`、`cross_attention`、`cdfi_branch`、可训练 encoder 层和文本投影层收集参数；删除对 `model.mlp_head` 和默认 CAM 对象的依赖。创建参数组后用参数 `id` 集合断言“无重复、无遗漏”：

```python
optimized = {
    id(parameter)
    for group in optimizer.param_groups
    for parameter in group["params"]
}
expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
if optimized != expected:
    raise RuntimeError("优化器参数组与可训练参数不一致")
```

训练循环每个 epoch：

```python
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
val_metrics = evaluate_loader(
    model,
    train_loader if args.task_mode == "overfit" else malignant_val_loader,
    args.task_mode,
    device,
    split_kind="malignant",
)
binary_val_metrics = None
if args.task_mode in {"flat5", "dual_head"}:
    binary_val_metrics = evaluate_loader(
        model,
        binary_val_loader,
        args.task_mode,
        device,
        split_kind="binary",
    )
score = (
    train_metrics["accuracy"]
    if args.task_mode == "overfit"
    else monitor_value(val_metrics, args.monitor_metric)
)
```

最佳模型保存 `model_state_dict`、epoch、`monitor_metric`、score、恶性验证指标和二分类验证指标；每轮更新 `history.json`，最佳轮更新 `metrics_best.json`。训练结束保存 `final_model.pth`，重新加载 `best_model.pth`，在固定恶性测试集以及适用的二分类测试集上评估，并将结果保存到 `metrics_test.json`。过拟合模式额外检查：

```python
if args.task_mode == "overfit":
    final_accuracy = history[-1]["train"]["accuracy"]
    final_loss = history[-1]["train"]["total_loss"]
    prediction_distribution = val_metrics["malignant"]["prediction_distribution"]
    passed = (
        final_accuracy >= 0.98
        and math.isfinite(final_loss)
        and all(count > 0 for count in prediction_distribution)
    )
    if not passed:
        raise RuntimeError(
            f"{args.overfit_samples} 例过拟合门禁失败："
            f"Accuracy={final_accuracy:.4f}，"
            f"loss={final_loss:.6f}，"
            f"prediction_distribution={prediction_distribution}"
        )
```

实现检查点确定性复评入口。它必须读取检查点相邻的 `args.json` 和 `split_manifest.json`，验证当前 `task_mode` 一致，按 case ID 恢复清单，而不是重新随机抽样：

```python
def restore_manifest_splits(canonical_splits, manifest):
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


def evaluate_checkpoint(args, device, canonical_splits):
    checkpoint_path = Path(args.evaluate_checkpoint)
    run_dir = checkpoint_path.parent
    saved_args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    if saved_args["task_mode"] != args.task_mode:
        raise ValueError("复评 task_mode 与检查点训练模式不一致")
    restored = restore_manifest_splits(canonical_splits, manifest)
    model_args = argparse.Namespace(**{**vars(args), **saved_args})
    model = build_model(model_args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.task_mode == "overfit":
        dataset = BUSOverfitDataset(
            PROJECT_ROOT,
            restored["train"],
            img_size=saved_args["img_size"],
        )
        metrics = evaluate_loader(
            model,
            build_eval_loader(dataset, args),
            "overfit",
            device,
            split_kind="malignant",
        )
    else:
        malignant_dataset = MultiModalBreastDataset(
            PROJECT_ROOT,
            split="test",
            img_size=saved_args["img_size"],
            max_text_len=saved_args["max_text_len"],
            samples=restored["malignant_test"],
            augment=False,
        )
        metrics = evaluate_loader(
            model,
            build_eval_loader(malignant_dataset, args),
            args.task_mode,
            device,
            split_kind="malignant",
        )
    save_json(Path(args.eval_output), metrics)
    return metrics
```

`build_model(args, device)` 和 `build_eval_loader(dataset, args)` 必须同时供训练与复评调用，避免两条路径出现模型参数或 DataLoader 配置漂移。`main(args)` 在创建新运行目录之前处理 `args.evaluate_checkpoint`，完成复评后立即返回。

- [ ] **步骤 4：更新忽略规则和实验记录模板**

在 `.gitignore` 追加：

```gitignore
# 本地实验运行产物
runs/
```

在 `docs/experiments/reliable-stage2-baselines.md` 写入：

```markdown
# Stage 2 可靠基线实验记录

## 数据契约

- 恶性训练/验证/测试：534/151/82，固定来自 `metadata.csv`。
- 额外良性训练/验证/测试：191/41/42，固定来自 `metadata_5class.csv`。
- 文件名基础组审计只作警告，不据此改动划分。

## 记录规则

- 门禁记录：运行目录、Seed、子集 case ID、最终训练损失、训练 Accuracy、预测分布和是否通过。
- 公平基线记录：运行目录、恶性 Macro-F1、Balanced Accuracy、四类 Recall、预测分布、logits 方差和特征方差。
- B1/B2 额外记录条件分型与端到端分型的差异；B2 记录良恶性 AUC、Sensitivity 和 Specificity。
- 所有数值只从对应目录的 `args.json`、`split_manifest.json`、`history.json`、`metrics_best.json` 和 `metrics_test.json` 读取。
```

- [ ] **步骤 5：运行 CLI 和全量单元测试**

```powershell
python -m pytest code/tests -q
```

预期：全部测试通过；需要真实模型权重的旧测试必须使用 `pytest.skip`，不能以打印后返回伪装通过。

- [ ] **步骤 6：运行 32 例过拟合门禁**

```powershell
python code/train/train_stage2.py --task_mode overfit --pretrained_path TinyUSFM.pth --overfit_samples 32 --no-augment --epochs 300 --batch_size 8 --lr 1e-3 --wd 0 --accum_steps 1 --seed 42
```

预期：

- 训练 Accuracy 至少 0.98，目标 1.00。
- 损失接近 0，且无 NaN/Inf。
- 预测分布包含四个亚型。
- `split_manifest.json` 固定记录 32 个 case ID，每类 8 例。

若失败，停止执行后续步骤；按顺序检查标签、logits 形状、损失输入、增强开关、冻结状态、优化器参数集合、梯度和 class-index 映射。

- [ ] **步骤 7：32 例通过后运行 64 例门禁**

```powershell
python code/train/train_stage2.py --task_mode overfit --pretrained_path TinyUSFM.pth --overfit_samples 64 --no-augment --epochs 400 --batch_size 8 --lr 1e-3 --wd 0 --accum_steps 1 --seed 42
```

预期：训练 Accuracy 至少 0.98，四类预测齐全；否则停止 B0/B1/B2。

- [ ] **步骤 8：记录门禁事实并提交统一入口**

将两个门禁运行目录中的最终损失、Accuracy、预测分布和运行目录名写入 `docs/experiments/reliable-stage2-baselines.md`，然后：

```powershell
git add .gitignore code/train/train_stage2.py code/tests/test_run_artifacts.py docs/experiments/reliable-stage2-baselines.md
git commit -m "feat: add reliable stage2 experiment entrypoint"
```

---

### 任务 9：运行公平 B0/B1/B2 并形成下一轮决策

**文件：**
- 修改：`docs/experiments/reliable-stage2-baselines.md`

**接口：**
- 输入：同一恶性 A/B/C、固定 seed 42、相同训练预算。
- 输出：B0/B1/B2 的恶性条件分型、恶性端到端分型、二分类与坍缩诊断结果。

- [ ] **步骤 1：运行 B0 平坦四分类**

```powershell
python code/train/train_stage2.py --task_mode flat4 --pretrained_path TinyUSFM.pth --epochs 100 --batch_size 8 --lr 5e-4 --wd 0.05 --unfreeze 2 --monitor_metric malignant_macro_f1 --seed 42
```

预期：产出恶性验证集 151 例的 Macro-F1、Balanced Accuracy、逐类 Recall、预测分布、logits 方差和特征方差。

- [ ] **步骤 2：运行 B1 平坦五分类诊断对照**

```powershell
python code/train/train_stage2.py --task_mode flat5 --pretrained_path TinyUSFM.pth --epochs 100 --batch_size 8 --lr 5e-4 --wd 0.05 --unfreeze 2 --monitor_metric malignant_macro_f1 --seed 42
```

预期：训练集为 534 恶性 + 191 良性；恶性验证仍为相同 151 例，同时报告条件四分类和把“预测良性”视为错误的端到端结果。

- [ ] **步骤 3：运行 B2 双头候选**

```powershell
python code/train/train_stage2.py --task_mode dual_head --pretrained_path TinyUSFM.pth --epochs 100 --batch_size 8 --lr 5e-4 --wd 0.05 --unfreeze 2 --lambda_bm 0.3 --monitor_metric malignant_macro_f1 --seed 42
```

预期：训练集与 B1 相同；报告恶性条件四分类、恶性端到端结果，以及良性+恶性验证集上的 AUC、Sensitivity 和 Specificity。

- [ ] **步骤 4：验证三个运行目录的划分清单**

用以下 PowerShell 自动选择最近一次 B0/B1/B2 运行目录并检查三个 `split_manifest.json`：

```powershell
$b0 = (Get-ChildItem 'runs\stage2' -Directory -Filter 'flat4_*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$b1 = (Get-ChildItem 'runs\stage2' -Directory -Filter 'flat5_*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$b2 = (Get-ChildItem 'runs\stage2' -Directory -Filter 'dual_head_*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$m0 = Get-Content "$b0\split_manifest.json" -Raw | ConvertFrom-Json
$m1 = Get-Content "$b1\split_manifest.json" -Raw | ConvertFrom-Json
$m2 = Get-Content "$b2\split_manifest.json" -Raw | ConvertFrom-Json
Compare-Object $m0.splits.malignant_val $m1.splits.malignant_val
Compare-Object $m0.splits.malignant_val $m2.splits.malignant_val
Compare-Object $m0.splits.malignant_test $m1.splits.malignant_test
Compare-Object $m0.splits.malignant_test $m2.splits.malignant_test
```

预期：四次 `Compare-Object` 都无输出。

- [ ] **步骤 5：重复评估最佳检查点以验证确定性**

以 B2 为例，自动选择最近一次 `dual_head` 运行目录，并连续复评同一检查点：

```powershell
$b2 = (Get-ChildItem 'runs\stage2' -Directory -Filter 'dual_head_*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
python code/train/train_stage2.py --task_mode dual_head --pretrained_path TinyUSFM.pth --evaluate_checkpoint "$b2\best_model.pth" --eval_output "$b2\metrics_eval_first.json" --seed 42
python code/train/train_stage2.py --task_mode dual_head --pretrained_path TinyUSFM.pth --evaluate_checkpoint "$b2\best_model.pth" --eval_output "$b2\metrics_eval_second.json" --seed 42
$firstHash = (Get-FileHash "$b2\metrics_eval_first.json").Hash
$secondHash = (Get-FileHash "$b2\metrics_eval_second.json").Hash
if ($firstHash -ne $secondHash) { throw '重复评估结果不一致' }
```

预期：两个 SHA256 完全一致；如不一致，检查评估增强、`model.eval()`、DataLoader 顺序和随机算子。

- [ ] **步骤 6：按明确标准决定是否采用 B2**

将以下判断写入 `docs/experiments/reliable-stage2-baselines.md`：

```text
B2 进入三 seed 验证，当且仅当：
1. 预测分布不再是单类坍缩；
2. 恶性 Macro-F1 同时高于 B0 和 B1 条件分型；
3. 提升不是仅来自良恶性 Accuracy；
4. HER2+ 与 TNBC Recall 没有出现不可接受的同步下降。
```

若 B2 满足单 seed 条件，依次运行 seed 3407、42、2026；只有至少两个 seed 保持同方向提升，才把双头架构作为当前 Stage 2 主方案。若不满足，先诊断 B0 的可学习性和各模态贡献，不引入动态权重或更复杂层级头。

- [ ] **步骤 7：提交实验事实，不提交运行目录**

```powershell
git add docs/experiments/reliable-stage2-baselines.md
git commit -m "docs: record fair stage2 baseline results"
```

提交前确认：

```powershell
git status --short
git diff --cached --stat
```

预期：暂存区只有轻量代码、测试和 Markdown；没有 `runs/`、`.pth` 或数据文件。

---

## 最终验证

- [ ] 运行完整测试：

```powershell
python -m pytest code/tests -q
```

预期：全部通过，无隐藏网络下载，无硬编码 `/home/lzj813/TinySpatial_Project` 路径。

- [ ] 检查中文输出和遗留硬编码：

```powershell
rg -n "/home/lzj813/TinySpatial_Project|mlp_head" code/tests code/train code/models/stage2 code/utils
rg -n "metadata_file" code/train/train_stage2.py
```

预期：两条命令均无输出，表示不存在旧 Linux 绝对路径、旧 `mlp_head` 依赖或训练入口中的旧单元数据参数。

- [ ] 检查计划范围外功能未混入：

```powershell
git diff --stat HEAD~9..HEAD
git status --short
```

预期：没有修改 Stage 1 算法、没有生成 ROI、没有新增动态模态权重、层级亚型树或更大 backbone。

- [ ] 检查当前分支和提交作者：

```powershell
git branch --show-current
git log -8 --format="%h %an <%ae> %s"
```

预期：分支为 `exp/reliable-stage2-baselines`；提交作者为 `LiZhijin-813 <543521673@qq.com>`。

- [ ] 停在本地审核点，不执行 `git push`。
