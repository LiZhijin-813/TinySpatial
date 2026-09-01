# 82例全测试集SWE置乱指标审计实施计划

> **供代理执行：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务实施；每一步使用复选框跟踪。

**目标：** 在已有检查点上，对完整 82 例恶性测试集执行原始 SWE 与确定性病例错配 SWE 的指标对照。

**架构：** 新建独立审计脚本，复用现有检查点恢复、测试集恢复、`SWEPerturbationDataset` 和 `evaluate_predictions`。脚本只生成病例记录与脱敏统计，不训练模型或保存大文件。

**技术栈：** Python、PyTorch、NumPy、pytest、现有数据集和评估工具。

## 全局约束

- 所有新增 Markdown、代码注释、文档字符串、命令行提示和错误信息必须使用中文。
- 固定使用检查点对应的完整 82 例恶性测试集。
- 只比较原始条件和 `case_shuffle` 条件；错配采用确定性循环后一位病例。
- 不训练、不修改标签、不保存权重、图像、原始文本或中间特征。
- 远端缓存与输出放在 `/data` 下，仓库不提交远端绝对路径。
- 不执行 GitHub 推送。

---

### Task 1：实现全测试集 SWE 置乱审计

**文件：**

- 创建：`code/train/swe_full_test_shuffle.py`
- 创建：`code/tests/test_swe_full_test_shuffle.py`

**接口：**

- 消费：`code.train.swe_stability_audit.SWEPerturbationDataset`、`code.train.model_evidence_audit.compare_prediction_records`、`code.utils.evaluation.evaluate_predictions`。
- 产生：`build_full_shuffle_indices(case_ids)`、`summarize_condition_metrics(rows)`、`run_audit(args)` 和命令行入口。

- [ ] **步骤 1：先写纯函数失败测试**

测试至少覆盖：

```python
import pytest

from code.train.swe_full_test_shuffle import (
    build_full_shuffle_indices,
    summarize_condition_metrics,
)


def test_full_shuffle_is_deterministic_and_has_no_self_pairing():
    case_ids = ["a", "b", "c"]
    indices = build_full_shuffle_indices(case_ids)
    assert indices == [1, 2, 0]
    assert all(index != position for position, index in enumerate(indices))


def test_full_shuffle_rejects_single_case():
    with pytest.raises(ValueError, match="至少需要两个病例"):
        build_full_shuffle_indices(["a"])


def test_metrics_keep_fixed_four_class_shape():
    rows = [
        {"true_label": 0, "base_predicted_label": 0, "shuffled_predicted_label": 1},
        {"true_label": 1, "base_predicted_label": 1, "shuffled_predicted_label": 1},
    ]
    summary = summarize_condition_metrics(rows)
    assert len(summary["base"]["confusion_matrix"]) == 4
    assert len(summary["shuffled"]["confusion_matrix"]) == 4
    assert len(summary["base"]["per_class_f1"]) == 4
```

- [ ] **步骤 2：运行测试确认按预期失败**

运行：

```powershell
& 'D:\Program\Anoconda\envs\yolov8\python.exe' -m pytest code/tests/test_swe_full_test_shuffle.py -q
```

预期：因审计脚本尚不存在而在收集阶段失败。

- [ ] **步骤 3：实现最小审计脚本**

实现固定的循环错配索引、指标汇总和检查点推理。`summarize_condition_metrics` 使用 `evaluate_predictions`，并固定 `class_names=["Luminal A", "Luminal B", "HER2+", "TNBC"]`；从结果中提取 Accuracy、Macro-F1、Balanced Accuracy、逐类 F1、混淆矩阵和预测分布。

`run_audit(args)` 必须：

1. 加载 `args.json`、`split_manifest.json` 和 `best_model.pth`；
2. 恢复完整 `malignant_test`，确认病例数为 82；
3. 对原始数据预测一次，对 `case_shuffle` 数据预测一次；
4. 按病例编号对齐两种预测并记录预测改变、置信度变化和供体标签；
5. 写出 `swe_full_test_cases.csv`、`swe_full_test_summary.json` 和中文 `swe_full_test_report.md`；
6. 记录数不是 82 时抛出中文错误。

命令行参数固定为：`--project_root`、`--run_dir`、`--output_dir`、`--device` 和 `--batch_size`。

- [ ] **步骤 4：运行定向测试、语法检查和差异检查**

运行：

```powershell
& 'D:\Program\Anoconda\envs\yolov8\python.exe' -m pytest code/tests/test_swe_full_test_shuffle.py code/tests/test_swe_stability_audit.py -q
& 'D:\Program\Anoconda\envs\yolov8\python.exe' -c "from pathlib import Path; compile(Path('code/train/swe_full_test_shuffle.py').read_text(encoding='utf-8'), 'code/train/swe_full_test_shuffle.py', 'exec'); print('语法检查通过')"
git diff --check
```

- [ ] **步骤 5：提交实现**

```powershell
git add code/train/swe_full_test_shuffle.py code/tests/test_swe_full_test_shuffle.py
git -c user.name='LiZhijin-813' -c user.email='543521673@qq.com' commit -m '新增全测试集SWE置乱审计'
```

### Task 2：远端运行并记录结果

**文件：**

- 创建：`docs/experiments/2026-09-01-swe-full-test-shuffle-results.md`
- 远端生成：`/data` 下的审计输出目录

**接口：**

- 消费：Task 1 的命令行入口和三个输出文件。
- 产生：原始与 SWE 错配的完整指标对照，以及一个明确的设计决策。

- [ ] **步骤 1：同步代码并运行远端定向测试**

远端使用现有 `tinyusfm` 环境；若 `cuda:0` 忙碌，改用空闲 GPU，只调整设备编号，不改变实验协议。

- [ ] **步骤 2：运行 82 例审计并核对输出**

确认两个条件均为 82 例，病例 CSV 为 82 条数据记录，并且输出目录只包含轻量 CSV、JSON 和 Markdown。

- [ ] **步骤 3：写入脱敏结果文档并提交**

结果文档必须包含原始和错配条件的 Accuracy、Macro-F1、Balanced Accuracy、四类 F1、混淆矩阵、预测改变率，以及基于结果的单一首选决策。不得写入远端绝对路径。

```powershell
git diff --check
git add docs/experiments/2026-09-01-swe-full-test-shuffle-results.md
git -c user.name='LiZhijin-813' -c user.email='543521673@qq.com' commit -m '记录全测试集SWE置乱结果'
```
