# BUS-SWE 归一化坐标对齐 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 允许原始尺寸不同的同病例 BUS/SWE 在共享归一化坐标系下接受训练和确定性评估，从而解除 B0/B1/B2 的数据入口阻断。

**Architecture:** 训练变换先将两模态分别缩放至同一公共方形画布，再用完全共享的裁剪、翻转和旋转参数生成输出。评估变换将两模态分别确定性缩放至同一 `img_size×img_size` 网格，不再把导出像素尺寸差异误判为病例错配。

**Tech Stack:** Python 3.12、PyTorch、torchvision transforms、Pillow、pytest。

## Global Constraints

- 不生成 ROI，不删除原始尺寸不同病例，不改元数据划分。
- BUS/SWE 保持共享空间变换；CDFI 保持独立变换。
- 评估阶段必须确定性；训练阶段应在固定随机种子下可复现。
- 所有注释、docstring、日志和 Markdown 使用中文。
- 只处理分辨率归一化，不宣称或伪造原始像素级配准。

---

### Task 1: 归一化坐标配对变换

**Files:**
- Modify: `code/datasets/dataset.py:50-130`
- Test: `code/tests/test_dataset.py`

**Interfaces:**
- Consumes: `PairedAlignedTransform(img_size).__call__(bus_img, swe_img)` 与 `PairedEvaluationTransform(img_size).__call__(bus_img, swe_img)`。
- Produces: 两张尺寸恒为 `(img_size, img_size)` 的 `PIL.Image.Image`，可直接由 `MultiModalBreastDataset` 的既有归一化流程消费。

- [ ] **Step 1: 写入失败测试**

将现有拒绝测试替换为以下两个行为测试：

```python
def test_evaluation_transform_normalizes_mismatched_paired_sizes():
    bus = Image.new("RGB", (310, 380), color=(128, 0, 0))
    swe = Image.new("RGB", (368, 372), color=(0, 0, 128))
    paired = PairedEvaluationTransform(img_size=224)
    first_bus, first_swe = paired(bus, swe)
    second_bus, second_swe = paired(bus, swe)
    assert first_bus.size == first_swe.size == (224, 224)
    assert first_bus.tobytes() == second_bus.tobytes()
    assert first_swe.tobytes() == second_swe.tobytes()


def test_training_transform_normalizes_mismatched_paired_sizes():
    bus = Image.new("RGB", (310, 380), color=(128, 0, 0))
    swe = Image.new("RGB", (368, 372), color=(0, 0, 128))
    bus_out, swe_out = PairedAlignedTransform(img_size=224)(bus, swe)
    assert bus_out.size == swe_out.size == (224, 224)
```

- [ ] **Step 2: 确认测试为红**

运行：

```powershell
python -m pytest code/tests/test_dataset.py -q -W error
```

预期：评估测试因 `BUS/SWE 原始尺寸不一致` 抛出 `ValueError`；训练测试因当前实现仅按 BUS 尺寸生成共享裁剪参数而无法安全处理不同尺寸。

- [ ] **Step 3: 最小化实现公共画布归一化**

在 `PairedAlignedTransform` 中加入私有方法：

```python
def _resize_to_square(self, image):
    return TF.resize(
        image,
        [self.img_size, self.img_size],
        antialias=True,
    )
```

训练变换开头分别调用该方法：

```python
bus_img = self._resize_to_square(bus_img)
swe_img = self._resize_to_square(swe_img)
```

随后保留共享 `RandomCrop.get_params`、翻转和旋转；当输入已是 `img_size×img_size` 时共享裁剪仍合法。

将 `PairedEvaluationTransform.__call__` 的尺寸断言替换为：

```python
bus_img = TF.resize(bus_img, [self.img_size, self.img_size], antialias=True)
swe_img = TF.resize(swe_img, [self.img_size, self.img_size], antialias=True)
return bus_img, swe_img
```

同步把两个类的 docstring 改为“归一化坐标下共享空间变换”，不要使用“像素级对应”或“原始尺寸相同”表述。

- [ ] **Step 4: 确认测试为绿**

运行：

```powershell
python -m pytest code/tests/test_dataset.py -q -W error
```

预期：所有数据集变换测试通过，且无 warning。

- [ ] **Step 5: 扩展真实入口回归测试**

在 `code/tests/test_dataset.py` 增加一个 `MultiModalBreastDataset` 显式样本测试：将 fixture 中一个 SWE 图重写为 `(368, 372)`，使用 `augment=False` 读取该病例，并断言 BUS/SWE 张量形状分别为 `(1, 224, 224)` 与 `(3, 224, 224)`。

- [ ] **Step 6: 运行完整测试并提交**

运行：

```powershell
python -m pytest code/tests -q -W error
git diff --check
```

预期：全部测试通过、无 warning、无空白错误。

提交：

```powershell
git add code/datasets/dataset.py code/tests/test_dataset.py
git commit -m "fix: normalize BUS-SWE paired coordinates"
```

### Task 2: 远端公平基线恢复与结果审计

**Files:**
- Modify: `docs/experiments/reliable-stage2-baselines.md`

**Interfaces:**
- Consumes: Task 1 提交后的训练入口，以及远端 `runs/stage2/{flat4_*,flat5_*,dual_head_*}` 运行产物。
- Produces: 记录 B0/B1/B2 的真实指标、划分一致性和 B2 决策结论的实验文档。

- [ ] **Step 1: 运行远端完整测试**

运行：

```bash
/home/lzj813/miniconda3/envs/tinyusfm/bin/python -m pytest code/tests -q -W error
```

预期：全部通过。

- [ ] **Step 2: 并行重跑 B0/B1/B2**

在三张空闲 GPU 上分别运行：

```bash
CUDA_VISIBLE_DEVICES=0 python code/train/train_stage2.py --task_mode flat4 --pretrained_path TinyUSFM.pth --epochs 100 --batch_size 8 --lr 5e-4 --wd 0.05 --unfreeze 2 --monitor_metric malignant_macro_f1 --seed 42 --device cuda:0
CUDA_VISIBLE_DEVICES=1 python code/train/train_stage2.py --task_mode flat5 --pretrained_path TinyUSFM.pth --epochs 100 --batch_size 8 --lr 5e-4 --wd 0.05 --unfreeze 2 --monitor_metric malignant_macro_f1 --seed 42 --device cuda:0
CUDA_VISIBLE_DEVICES=2 python code/train/train_stage2.py --task_mode dual_head --pretrained_path TinyUSFM.pth --epochs 100 --batch_size 8 --lr 5e-4 --wd 0.05 --unfreeze 2 --lambda_bm 0.3 --monitor_metric malignant_macro_f1 --seed 42 --device cuda:0
```

- [ ] **Step 3: 验证公平划分与 B2 确定性**

比较三个 `split_manifest.json` 中的 `malignant_val` 与 `malignant_test` case ID；要求完全一致。对 B2 最佳检查点连续复评两次，比较输出 JSON 的 SHA256；要求完全一致。

- [ ] **Step 4: 记录结果并作出 B2 决策**

在实验文档记录 B0/B1/B2 的恶性条件 Macro-F1、Balanced Accuracy、四类 Recall、预测分布、logits/特征方差；B1/B2 记录端到端恶性结果；B2 记录二分类 AUC、Sensitivity、Specificity。

只有同时满足以下条件时，安排 B2 三 seed 验证：预测分布未坍缩；恶性 Macro-F1 高于 B0 与 B1 条件分型；提升不只来自良恶性 Accuracy；HER2+ 与 TNBC Recall 没有不可接受的同步下降。

- [ ] **Step 5: 提交实验事实**

```powershell
git add docs/experiments/reliable-stage2-baselines.md
git commit -m "docs: record normalized-coordinate baselines"
```
