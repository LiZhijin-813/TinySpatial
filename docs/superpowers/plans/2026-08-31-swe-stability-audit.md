# SWE 稳定性审计实施计划

> **供代理执行：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务实施；每一步使用复选框跟踪。

**目标：** 在已有 `flat5` 检查点上，对固定 36 个病例执行四种确定性 SWE 扰动，生成可复核的病例级记录和审计结论。

**架构：** 新建独立的 `swe_stability_audit.py`，用轻量数据集包装器只修改 SWE 张量，并复用现有模型证据审计中的检查点恢复、病例选择和预测比较逻辑。先以纯函数测试锁定张量扰动、错配映射和指标口径，再在远端 `tinyusfm` 环境运行完整推理；不修改原训练入口。

**技术栈：** Python、PyTorch、NumPy、pytest、现有 `MultiModalBreastDataset` 与 Stage 2 检查点加载接口。

## 全局约束

- 所有新增 Markdown、代码注释、文档字符串、命令行提示和错误信息必须使用中文。
- 固定使用 36 个病例：24 个基线错误病例和 12 个基线正确病例。
- 固定四个条件：`intensity_0_9`、`intensity_1_1`、`horizontal_flip`、`case_shuffle`。
- 共生成 144 条扰动记录；不自动扩大病例数或扰动集合。
- 不训练、不修改标签、不保存模型权重、扰动图像、原始文本或中间特征张量。
- 远端输出写入 `/data/lzj813/b1-data-evidence-audit-20260811/swe-stability`。
- 只提交当前任务文件，不处理三个历史未跟踪报告，不执行 GitHub 推送。

---

### Task 1：实现 SWE 扰动、预测和汇总

**文件：**

- 创建：`code/train/swe_stability_audit.py`
- 创建：`code/tests/test_swe_stability_audit.py`

**接口：**

- 消费：`code.train.model_evidence_audit._load_json`、`_select_case_ids` 和 `compare_prediction_records`。
- 产生：`apply_swe_perturbation(swe_tensor, condition)`、`build_shuffle_indices(case_ids)`、`summarize_stability(rows)`、`run_audit(args)` 和命令行入口。

- [ ] **步骤 1：先写纯函数失败测试**

在 `code/tests/test_swe_stability_audit.py` 中覆盖以下行为：

```python
import pytest
import torch

from code.train.swe_stability_audit import (
    apply_swe_perturbation,
    build_shuffle_indices,
    summarize_stability,
)


def test_intensity_perturbation_uses_image_range_and_clamps():
    swe = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    result = apply_swe_perturbation(swe, "intensity_1_1")
    assert result[0, 0, 0].tolist() == pytest.approx([-1.0, 0.1, 1.0])


def test_horizontal_flip_only_reverses_width():
    swe = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    result = apply_swe_perturbation(swe, "horizontal_flip")
    assert result.tolist() == [[[[3.0, 2.0, 1.0]]]]


def test_shuffle_indices_are_deterministic_derangement():
    assert build_shuffle_indices(["a", "b", "c"]) == [1, 2, 0]


def test_shuffle_rejects_single_case():
    with pytest.raises(ValueError, match="至少需要两个病例"):
        build_shuffle_indices(["a"])


def test_summary_counts_damage_correction_and_donor_agreement():
    rows = [
        {
            "condition": "case_shuffle",
            "base_correct": True,
            "perturbed_correct": False,
            "prediction_changed": True,
            "confidence_delta": -0.2,
            "perturbed_predicted_label": 2,
            "donor_label": 2,
        },
        {
            "condition": "case_shuffle",
            "base_correct": False,
            "perturbed_correct": True,
            "prediction_changed": True,
            "confidence_delta": 0.1,
            "perturbed_predicted_label": 1,
            "donor_label": 3,
        },
    ]
    summary = summarize_stability(rows)["case_shuffle"]
    assert summary["prediction_changed_count"] == 2
    assert summary["correct_damaged_count"] == 1
    assert summary["error_corrected_count"] == 1
    assert summary["donor_agreement_rate"] == pytest.approx(0.5)
```

- [ ] **步骤 2：运行测试并确认按预期失败**

运行：

```powershell
& 'D:\Program\Anocanda\envs\yolov8\python.exe' -m pytest code/tests/test_swe_stability_audit.py -q
```

预期：测试收集阶段因 `code.train.swe_stability_audit` 尚不存在而失败。

- [ ] **步骤 3：实现最小纯函数和数据集包装器**

在 `code/train/swe_stability_audit.py` 中实现：

```python
CONDITIONS = (
    "intensity_0_9",
    "intensity_1_1",
    "horizontal_flip",
    "case_shuffle",
)


def apply_swe_perturbation(swe_tensor, condition):
    if condition == "horizontal_flip":
        return torch.flip(swe_tensor, dims=(-1,))
    factors = {"intensity_0_9": 0.9, "intensity_1_1": 1.1}
    if condition not in factors:
        raise ValueError(f"未知 SWE 扰动条件：{condition}")
    pixels = ((swe_tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    return (pixels.mul(factors[condition]).clamp(0.0, 1.0) * 2.0) - 1.0


def build_shuffle_indices(case_ids):
    if len(case_ids) < 2:
        raise ValueError("病例错配至少需要两个病例")
    return list(range(1, len(case_ids))) + [0]
```

新增 `SWEPerturbationDataset` 包装已有确定性评估数据集：普通条件只修改当前样本的 SWE；`case_shuffle` 使用循环后一位病例的 SWE，同时返回 `donor_case_id` 和 `donor_label`。包装器不得改变 BUS、CDFI、文本和标签。

- [ ] **步骤 4：实现预测、汇总和命令行入口**

`run_audit(args)` 必须：

1. 加载 `args.json`、`split_manifest.json` 和 `best_model.pth`；
2. 复用 `_select_case_ids` 选择固定 36 例；
3. 为完整输入生成一次基线预测；
4. 为四个条件各生成 36 条记录；
5. 使用 `compare_prediction_records` 计算预测变化和置信度变化；
6. 补充 `base_correct`、`perturbed_correct`、`donor_case_id` 和 `donor_label`；
7. 写出 `swe_stability_cases.csv`、`swe_stability_summary.json` 和中文 `swe_stability_report.md`；
8. 若记录数不等于 `len(case_ids) * 4`，抛出中文错误。

`summarize_stability(rows)` 按条件输出：`row_count`、`prediction_changed_count`、`prediction_changed_rate`、`mean_confidence_delta`、`base_correct_count`、`correct_damaged_count`、`correct_damaged_rate`、`base_error_count`、`error_corrected_count` 和 `error_corrected_rate`；仅 `case_shuffle` 增加 `donor_agreement_count` 与 `donor_agreement_rate`。

命令行参数固定为：

```text
--project_root
--run_dir
--case_audit_dir
--quality_flags
--output_dir
--device
--max_error_cases 24
--max_correct_cases 12
--batch_size 8
```

- [ ] **步骤 5：运行定向测试、语法检查和差异检查**

运行：

```powershell
& 'D:\Program\Anocanda\envs\yolov8\python.exe' -m pytest code/tests/test_swe_stability_audit.py code/tests/test_model_evidence_audit.py -q
& 'D:\Program\Anocanda\envs\yolov8\python.exe' -c "from pathlib import Path; compile(Path('code/train/swe_stability_audit.py').read_text(encoding='utf-8'), 'code/train/swe_stability_audit.py', 'exec'); print('语法检查通过')"
git diff --check
```

预期：所有定向测试通过，语法检查输出“语法检查通过”，差异检查无输出。

- [ ] **步骤 6：提交实现**

```powershell
git add code/train/swe_stability_audit.py code/tests/test_swe_stability_audit.py
git -c user.name='LiZhijin-813' -c user.email='543521673@qq.com' commit -m '新增SWE稳定性审计'
```

### Task 2：远端运行并记录实验结论

**文件：**

- 创建：`docs/experiments/2026-08-31-swe-stability-audit-results.md`
- 远端生成：`/data/lzj813/b1-data-evidence-audit-20260811/swe-stability/*`

**接口：**

- 消费：任务 1 的命令行入口及三个输出文件。
- 产生：脱敏实验记录和明确的下一步模型决策。

- [ ] **步骤 1：同步代码并运行远端定向测试**

```powershell
scp D:/Project/TinySpatial/code/train/swe_stability_audit.py anon-service2:/data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/train/swe_stability_audit.py
scp D:/Project/TinySpatial/code/tests/test_swe_stability_audit.py anon-service2:/data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/tests/test_swe_stability_audit.py
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python -m pytest code/tests/test_swe_stability_audit.py code/tests/test_model_evidence_audit.py -q"
```

预期：远端定向测试全部通过。

- [ ] **步骤 2：运行固定 36 例审计**

```powershell
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/swe_stability_audit.py --project_root /home/lzj813/TinySpatial_Project --run_dir /data/lzj813/b1-modality-ablation-20260809/full/flat5_20260809-112135 --case_audit_dir /data/lzj813/b1-case-audit-20260731-232334 --quality_flags /data/lzj813/b1-data-evidence-audit-20260811/data-quality/case_quality_flags.csv --output_dir /data/lzj813/b1-data-evidence-audit-20260811/swe-stability --device cuda:0 --max_error_cases 24 --max_correct_cases 12 --batch_size 8"
```

预期：生成 144 条病例扰动记录，且不产生模型权重和扰动图像。

- [ ] **步骤 3：核对结果完整性和目录大小**

```powershell
ssh anon-service2 "find /data/lzj813/b1-data-evidence-audit-20260811/swe-stability -maxdepth 1 -type f -printf '%f\n' | sort && du -sh /data/lzj813/b1-data-evidence-audit-20260811/swe-stability"
```

必须存在：`swe_stability_cases.csv`、`swe_stability_summary.json` 和 `swe_stability_report.md`。CSV 必须为 144 行数据记录，不计表头。

- [ ] **步骤 4：写入脱敏实验结果**

`docs/experiments/2026-08-31-swe-stability-audit-results.md` 必须包含：

- 四种条件的预测改变率、平均置信度变化、正确病例受损率和错误病例纠正率；
- `case_shuffle` 的供体标签一致率；
- 与 SWE 零屏蔽 75% 改变率的相对比较；
- 明确区分输入敏感性、空间对应和真实病灶定位；
- 根据规格中的判定规则给出一个首选下一步，不列出大量并行备选实验。

- [ ] **步骤 5：验证并提交结果文档**

```powershell
git diff --check
git add docs/experiments/2026-08-31-swe-stability-audit-results.md
git -c user.name='LiZhijin-813' -c user.email='543521673@qq.com' commit -m '记录SWE稳定性审计结果'
```

最终不得执行 `git push`。
