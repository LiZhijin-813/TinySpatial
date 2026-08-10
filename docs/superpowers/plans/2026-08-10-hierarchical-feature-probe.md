# 病例组五折层级特征探针实现计划

> **面向执行代理：** 必须按任务逐项执行；每个任务完成后运行对应测试并提交。实现阶段使用测试驱动开发。

**目标：** 新增一个基于病例组五折清单的层级二分类固定特征探针，用于定位四分类分型失败的具体决策层级。

**架构：** 在 `run_cv_hierarchical_probe.py` 中复用现有固定检查点特征提取流程和 `fit_linear_probe`。纯函数负责定义层级、筛选样本、映射标签和聚合指标；主流程负责加载检查点、按病例组折拟合二分类探针并保存 JSON。现有训练代码和四分类探针代码保持不变。

**技术栈：** Python、PyTorch、NumPy、pytest、现有 `evaluate_predictions`。

## 全局约束

- 所有新注释、文档字符串、命令行提示和错误信息使用中文。
- 每折只能使用训练折拟合探针，测试折只用于评估。
- `luminal_a_vs_luminal_b` 和 `her2_vs_tnbc` 必须排除无关类别，不能将无关类别当作负类。
- 只写入 `/data/lzj813/b1-modality-diagnosis-20260809/` 作为远端实验输出，不把权重写入仓库。
- 不处理或删除现有未跟踪目录 `.hf_cache/`。
- 不推送 GitHub。

---

### 任务 1：为层级标签纯函数编写失败测试

**文件：**
- 新增：`D:/Project/TinySpatial/code/tests/test_hierarchical_probe.py`
- 待新增：`D:/Project/TinySpatial/code/train/run_cv_hierarchical_probe.py`

**接口：**
- `build_hierarchical_tasks()` 返回三个任务配置字典，键为 `luminal_vs_non_luminal`、`luminal_a_vs_luminal_b`、`her2_vs_tnbc`。
- 每个配置包含 `positive_labels`、`negative_labels` 和 `display_names`。
- `select_hierarchical_labels(labels, task)` 接受一维整型张量，返回 `(mask, binary_labels)`。

- [ ] **步骤 1：写失败测试**

```python
import pytest
import torch

from code.train.run_cv_hierarchical_probe import (
    build_hierarchical_tasks,
    select_hierarchical_labels,
)


def test_luminal_internal_task_excludes_unrelated_classes():
    tasks = build_hierarchical_tasks()
    mask, binary = select_hierarchical_labels(
        torch.tensor([0, 1, 2, 3, 0]),
        tasks["luminal_a_vs_luminal_b"],
    )
    assert mask.tolist() == [True, True, False, False, True]
    assert binary.tolist() == [0, 1, 0]


def test_hierarchical_task_rejects_labels_outside_four_class_space():
    tasks = build_hierarchical_tasks()
    with pytest.raises(ValueError, match="标签"):
        select_hierarchical_labels(
            torch.tensor([0, 4]),
            tasks["luminal_vs_non_luminal"],
        )
```

- [ ] **步骤 2：运行测试确认按预期失败**

运行：

```powershell
python -m pytest code/tests/test_hierarchical_probe.py -q
```

预期：因 `run_cv_hierarchical_probe.py` 尚不存在而失败，不能是测试语法错误。

- [ ] **步骤 3：实现最小标签配置和筛选函数**

```python
def build_hierarchical_tasks():
    return {
        "luminal_vs_non_luminal": {
            "positive_labels": (0, 1),
            "negative_labels": (2, 3),
            "display_names": ("Luminal", "非Luminal"),
        },
        "luminal_a_vs_luminal_b": {
            "positive_labels": (0,),
            "negative_labels": (1,),
            "display_names": ("Luminal A", "Luminal B"),
        },
        "her2_vs_tnbc": {
            "positive_labels": (2,),
            "negative_labels": (3,),
            "display_names": ("HER2+", "TNBC"),
        },
    }


def select_hierarchical_labels(labels, task):
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("标签必须是一维张量")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("标签必须是整数张量")
    if torch.any((labels < 0) | (labels >= 4)):
        raise ValueError("标签超出四分类标签空间")
    positive = torch.zeros_like(labels, dtype=torch.bool)
    negative = torch.zeros_like(labels, dtype=torch.bool)
    for label in task["positive_labels"]:
        positive |= labels == label
    for label in task["negative_labels"]:
        negative |= labels == label
    mask = positive | negative
    binary_labels = torch.where(positive[mask], 0, 1)
    return mask, binary_labels
```

- [ ] **步骤 4：运行测试确认通过**

运行：`python -m pytest code/tests/test_hierarchical_probe.py -q`

预期：`2 passed`。

- [ ] **步骤 5：提交纯函数测试与实现**

```powershell
git add code/tests/test_hierarchical_probe.py code/train/run_cv_hierarchical_probe.py
git commit -m "新增层级探针标签筛选函数"
```

### 任务 2：增加指标聚合和折内有效样本校验

**文件：**
- 修改：`D:/Project/TinySpatial/code/train/run_cv_hierarchical_probe.py`
- 修改：`D:/Project/TinySpatial/code/tests/test_hierarchical_probe.py`

**接口：**
- `aggregate_binary_metrics(fold_metrics)` 返回 `macro_f1` 和 `balanced_accuracy` 的 `mean`、`std`、`values`。
- `validate_fold_labels(train_labels, query_labels, task_name)` 在任一集合缺少二分类类别时抛出中文 `ValueError`。

- [ ] **步骤 1：写失败测试并确认失败**

```python
from code.train.run_cv_hierarchical_probe import (
    aggregate_binary_metrics,
    validate_fold_labels,
)


def test_aggregate_binary_metrics_returns_mean_std_and_values():
    result = aggregate_binary_metrics([
        {"macro_f1": 0.2, "balanced_accuracy": 0.3},
        {"macro_f1": 0.4, "balanced_accuracy": 0.5},
    ])
    assert result["macro_f1"]["mean"] == pytest.approx(0.3)
    assert result["macro_f1"]["std"] == pytest.approx(0.1)
    assert result["balanced_accuracy"]["values"] == [0.3, 0.5]


def test_validate_fold_labels_rejects_missing_binary_class():
    with pytest.raises(ValueError, match="缺少"):
        validate_fold_labels(torch.tensor([0, 0]), torch.tensor([0, 1]), "测试层级")
```

运行：`python -m pytest code/tests/test_hierarchical_probe.py -q`

预期：因两个函数尚未实现而失败。

- [ ] **步骤 2：实现最小校验和聚合逻辑**

实现以下两个函数：

```python
def validate_fold_labels(train_labels, query_labels, task_name):
    for name, labels in (("训练折", train_labels), ("测试折", query_labels)):
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
            raise ValueError(f"{task_name}{name}标签必须是一维张量")
        if set(labels.tolist()) != {0, 1} and not {0, 1}.issubset(set(labels.tolist())):
            raise ValueError(f"{task_name}{name}缺少二分类中的某一类别")


def aggregate_binary_metrics(fold_metrics):
    if not fold_metrics:
        raise ValueError("层级指标不能为空")
    summary = {}
    for name in ("macro_f1", "balanced_accuracy"):
        values = np.asarray([item[name] for item in fold_metrics], dtype=float)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "values": [float(value) for value in values],
        }
    return summary
```

使用 `numpy.asarray` 计算 `mean` 和 `std(ddof=0)`，将结果转换为 Python `float`；校验训练折和查询折都同时包含标签 0、1。

- [ ] **步骤 3：运行测试确认通过**

运行：`python -m pytest code/tests/test_hierarchical_probe.py -q`

预期：`4 passed`。

- [ ] **步骤 4：提交**

```powershell
git add code/tests/test_hierarchical_probe.py code/train/run_cv_hierarchical_probe.py
git commit -m "增加层级探针指标聚合校验"
```

### 任务 3：实现固定检查点的病例组五折层级探针流程

**文件：**
- 修改：`D:/Project/TinySpatial/code/train/run_cv_hierarchical_probe.py`
- 修改：`D:/Project/TinySpatial/code/tests/test_hierarchical_probe.py`

**接口：**
- `run_cv_hierarchical_probe(run_dir, cv_manifest, output_dir, device_name="cuda:0", batch_size=16)` 返回结果字典并写出 `hierarchical_probe_metrics.json`。
- 命令行参数沿用 `run_cv_feature_probe.py`：`--run_dir`、`--cv_manifest`、`--output_dir`、`--device`、`--batch_size`。

- [ ] **步骤 1：增加纯流程边界测试**

测试非法清单折、空层级筛选结果和 `flat4` 检查点必须抛出中文 `ValueError`；测试不加载 GPU 权重。

- [ ] **步骤 2：运行边界测试确认失败**

运行：`python -m pytest code/tests/test_hierarchical_probe.py -q`

预期：新增边界测试因流程函数尚未完成而失败。

- [ ] **步骤 3：实现流程**

复用现有 `_read_json`、`_resolve_device`、`_collect_features`，或者在新脚本中通过显式导入复用；读取 `args.json`、`best_model.pth`、病例元数据和五折清单。对每个任务和每个折：

```python
train_mask, train_binary = select_hierarchical_labels(train_labels, task)
query_mask, query_binary = select_hierarchical_labels(query_labels, task)
validate_fold_labels(train_binary, query_binary, task_name)
counts = torch.bincount(train_binary, minlength=2).float()
weights = counts.sum() / (2.0 * counts)
plain = fit_linear_probe(
    train_features[train_mask], train_binary, query_features[query_mask],
    num_classes=2, epochs=300, learning_rate=0.05, weight_decay=1e-4,
)
inverse = fit_linear_probe(
    train_features[train_mask], train_binary, query_features[query_mask],
    num_classes=2, epochs=300, learning_rate=0.05, weight_decay=1e-4,
    class_weights=weights,
)
```

保存每折的有效样本数、标签分布、预测分布、混淆矩阵、召回率和两种方法的完整评估指标。

- [ ] **步骤 4：运行完整本地测试**

运行：`python -m pytest code/tests -q`

预期：现有测试和新增测试全部通过，数量不低于当前 `254` 个测试加新增测试。

- [ ] **步骤 5：提交实现**

```powershell
git add code/tests/test_hierarchical_probe.py code/train/run_cv_hierarchical_probe.py
git commit -m "新增病例组五折层级特征探针"
```

### 任务 4：远端集成验证和三检查点诊断

**文件：**
- 远端输出：`/data/lzj813/b1-modality-diagnosis-20260809/hierarchical-probe-*`
- 代码不再修改，除非集成测试暴露协议错误。

- [ ] **步骤 1：同步当前本地提交到远端实验副本**

只同步已提交代码到 `/data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H`，不上传权重和 `.hf_cache/`：

```powershell
scp D:/Project/TinySpatial/code/train/run_cv_hierarchical_probe.py anon-service2:/data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/train/run_cv_hierarchical_probe.py
scp D:/Project/TinySpatial/code/tests/test_hierarchical_probe.py anon-service2:/data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/tests/test_hierarchical_probe.py
```

同步后运行：

```bash
cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H
/home/lzj813/miniconda3/envs/tinyusfm/bin/python -m pytest code/tests/test_hierarchical_probe.py -q
```

预期：新增测试全部通过。

- [ ] **步骤 2：运行 BUS+Text 固定检查点**

```bash
/home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/run_cv_hierarchical_probe.py \
  --run_dir /data/lzj813/b1-modality-diagnosis-20260809/bus-text-no-weight-unfreeze0/flat5_20260810-000014 \
  --cv_manifest /data/lzj813/b1-modality-diagnosis-20260809/cv-manifest-group5-seed42.json \
  --output_dir /data/lzj813/b1-modality-diagnosis-20260809/hierarchical-probe-bus-text-unfreeze0 \
  --device cuda:0 --batch_size 16
```

- [ ] **步骤 3：运行默认全模态固定检查点**

使用同一命令，将检查点替换为 `/data/lzj813/b1-modality-diagnosis-20260809/full-patience50/flat5_20260809-205124`，输出目录替换为 `hierarchical-probe-full-patience50`。

- [ ] **步骤 4：读取结果并做层级解释**

比较三个层级的五折均值和标准差；若大类任务明显高于内部任务，优先考虑分层分类结构；若三者均低，记录为固定融合表示整体分型信号不足，不继续盲目调损失。

- [ ] **步骤 5：完成验证记录**

记录远端 JSON 路径、样本数、每层级两种探针指标，并在本地提交验证说明；不推送 GitHub。
