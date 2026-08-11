# 数据质量与模型证据审计实施计划

> **面向执行代理：** 必须按任务逐项执行；推荐使用 `superpowers:subagent-driven-development`，每个任务完成后独立审查。步骤使用复选框跟踪。

**目标：** 在不重新训练模型的前提下，识别标签/文本/模态质量问题，以及模型是否依赖不稳定的空间或文本证据。

**架构：** 新增一个全量数据质量审计脚本和一个有限病例模型证据审计脚本。前者只读取元数据、图像文件和文本 JSON，后者复用现有模型加载、病例审计和模态屏蔽接口，对同一检查点做预测扰动和梯度显著性分析。两者均输出脱敏统计，不改变训练流程和模型结构。

**技术栈：** Python、PyTorch、Pillow、NumPy、pytest、现有 `split_utils.py`、`audit_stage2.py`、`train_stage2.py` 和远端 `tinyusfm` 环境。

## 全局约束

- 所有新增注释、文档字符串、命令行提示、错误信息和报告使用中文。
- 不修改标签、不修改原始运行目录、不重新训练、不新增模态组合。
- 原始文本只在远端 `/data` 临时读取；Git 只提交脱敏统计和结论。
- 不提交权重、缓存、原始文本、完整显著性张量或训练日志。
- 任务一完成后，若发现标签冲突、跨切分重复或明显文本泄漏，任务二只保留必要的确认性检查，不扩展模型实验。
- 任务二最多分析 `36` 个病例，其中最多 `24` 个错误病例和 `12` 个正确病例；最多保存 `12` 张可视化图。
- 所有远端输出写入 `/data/lzj813/b1-data-evidence-audit-20260811`。
- 不推送 GitHub。

## 文件与职责

- 创建：`code/train/data_quality_audit.py`，负责全量数据质量和文本风险统计。
- 创建：`code/tests/test_data_quality_audit.py`，覆盖数据审计纯函数。
- 创建：`code/train/model_evidence_audit.py`，负责有限病例预测扰动和空间证据统计。
- 创建：`code/tests/test_model_evidence_audit.py`，覆盖证据汇总、文本遮蔽和显著性统计纯函数。
- 创建：`docs/experiments/2026-08-11-data-quality-audit-results.md`。
- 创建：`docs/experiments/2026-08-11-model-evidence-audit-results.md`。

## Task 1：实现并执行全量数据质量审计

**文件：**

- 创建：`code/train/data_quality_audit.py`
- 创建：`code/tests/test_data_quality_audit.py`
- 读取：`code/datasets/split_utils.py`
- 读取：`code/train/case_audit.py`

**接口：**

```python
def classify_text_risk(raw_text: str) -> dict:
    """返回文本长度、空文本状态和目标关键词命中情况。"""

def audit_image_file(path: Path) -> dict:
    """返回文件可读性、尺寸、通道数和近似常量图像标记。"""

def build_quality_summary(samples: list[dict], quality_rows: list[dict]) -> dict:
    """汇总标签、病例组、模态完整性和文本风险统计。"""

def write_quality_outputs(output_dir: Path, summary: dict, rows: list[dict]) -> None:
    """写入脱敏 JSON、CSV 和中文 Markdown 报告。"""
```

- [ ] **步骤 1：编写失败测试**

测试必须覆盖：空文本、包含 `Luminal B` 和 `TNBC` 的文本、可读取图像、近似常量图像、缺失图像、重复病例组和标签计数汇总。

```python
def test_classify_text_risk_marks_target_terms():
    result = classify_text_risk("报告提示 Luminal B，未见 TNBC")
    assert result["target_keyword_hits"] == ["Luminal B", "TNBC"]
    assert result["contains_target_keyword"] is True

def test_audit_image_file_marks_missing_file(tmp_path):
    result = audit_image_file(tmp_path / "missing.png")
    assert result["exists"] is False
    assert result["readable"] is False

def test_build_quality_summary_counts_labels_and_risks():
    rows = [
        {"case_id": "a", "subtype_label": 0, "group_id": "a", "bus_exists": True,
         "swe_exists": True, "cdfi_exists": True, "text_exists": True,
         "contains_target_keyword": False, "label_conflict": False},
        {"case_id": "b", "subtype_label": 1, "group_id": "a", "bus_exists": True,
         "swe_exists": False, "cdfi_exists": True, "text_exists": True,
         "contains_target_keyword": True, "label_conflict": True},
    ]
    summary = build_quality_summary(rows, rows)
    assert summary["label_counts"] == [1, 1, 0, 0]
    assert summary["multi_case_group_count"] == 1
    assert summary["target_keyword_case_count"] == 1
    assert summary["label_conflict_case_count"] == 1
```

- [ ] **步骤 2：运行失败测试**

运行：

```powershell
python -m pytest code/tests/test_data_quality_audit.py -q
```

预期：因 `data_quality_audit.py` 尚不存在而失败。

- [ ] **步骤 3：实现最小审计逻辑**

目标关键词固定为 `Luminal A`、`Luminal B`、`HER2`、`TNBC` 及其大小写不敏感形式；关键词命中只做风险标记。图像审计使用 Pillow 读取，计算宽度、高度、通道数、像素标准差和 `near_constant` 标记；读取失败只记录错误类型，不抛出单病例级异常。

病例组 ID 必须复用 `derive_suspected_group_id`，标签统计只接受 `0` 到 `3` 的恶性亚型。输出行只能包含病例 ID、标签、组 ID、文件质量和风险布尔值，不写入原始文本。

命令行参数固定为：

```text
--project_root
--metadata_file
--case_audit_dir
--output_dir
```

- [ ] **步骤 4：运行测试并提交代码**

```powershell
python -m pytest code/tests/test_data_quality_audit.py -q
git add code/train/data_quality_audit.py code/tests/test_data_quality_audit.py
git commit -m "新增全量数据质量审计"
```

预期：新增测试全部通过，提交不包含数据和审计产物。

- [ ] **步骤 5：远端执行全量审计**

```powershell
scp D:/Project/TinySpatial/code/train/data_quality_audit.py anon-service2:/data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/train/data_quality_audit.py
scp D:/Project/TinySpatial/code/tests/test_data_quality_audit.py anon-service2:/data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/tests/test_data_quality_audit.py
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python -m pytest code/tests/test_data_quality_audit.py -q"
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/data_quality_audit.py --project_root /home/lzj813/TinySpatial_Project --metadata_file metadata.csv --case_audit_dir /data/lzj813/b1-case-audit-20260731-232334 --output_dir /data/lzj813/b1-data-evidence-audit-20260811/data-quality"
```

预期：生成 `data_quality_summary.json`、`case_quality_flags.csv` 和 `data_quality_report.md`；Git 中不出现原始文本。

## Task 2：实现并执行模型证据审计

**文件：**

- 创建：`code/train/model_evidence_audit.py`
- 创建：`code/tests/test_model_evidence_audit.py`
- 复用：`code/train/audit_stage2.py`
- 复用：`code/train/train_stage2.py`
- 复用：`code/datasets/dataset.py`

**接口：**

```python
def mask_text_terms(raw_text: str, terms: list[str]) -> str:
    """用统一遮蔽标记替换指定文本关键词。"""

def compare_prediction_records(base_rows: list[dict], ablated_rows: list[dict]) -> list[dict]:
    """按病例 ID 对齐完整输入与模态屏蔽预测，返回预测变化记录。"""

def compute_spatial_concentration(saliency: np.ndarray, center_fraction: float = 0.6) -> dict:
    """计算中心区域、边缘区域和总显著性比例。"""

def summarize_evidence(rows: list[dict]) -> dict:
    """按标签、错误类型和模态汇总预测与空间证据变化。"""
```

- [ ] **步骤 1：编写失败测试**

```python
import pytest

def test_mask_text_terms_replaces_only_requested_terms():
    result = mask_text_terms("Luminal B，HER2 阴性", ["Luminal B"])
    assert result == "[遮蔽]，HER2 阴性"

def test_compute_spatial_concentration_returns_center_and_edge_ratio():
    saliency = np.zeros((4, 4), dtype=np.float32)
    saliency[1:3, 1:3] = 1.0
    result = compute_spatial_concentration(saliency, center_fraction=0.5)
    assert result["center_ratio"] == 1.0
    assert result["edge_ratio"] == 0.0

def test_compare_prediction_records_marks_changed_case():
    base = [{"case_id": "a", "predicted_label": 0, "confidence": 0.8}]
    ablated = [{"case_id": "a", "predicted_label": 2, "confidence": 0.6}]
    result = compare_prediction_records(base, ablated)
    assert result[0]["prediction_changed"] is True
    assert result[0]["confidence_delta"] == pytest.approx(-0.2)
```

- [ ] **步骤 2：运行失败测试**

```powershell
python -m pytest code/tests/test_model_evidence_audit.py -q
```

预期：因 `model_evidence_audit.py` 尚不存在而失败。

- [ ] **步骤 3：实现最小证据审计逻辑**

预测扰动部分复用 `audit_stage2.py` 的检查点加载和数据集构造，不复制模型恢复逻辑。对完整输入、屏蔽 SWE、屏蔽 CDFI、屏蔽 Text 分别生成病例预测记录；文本关键词遮蔽只对任务一标记为有目标关键词的重点病例执行。

空间证据部分将模型置于 `eval()`，不创建优化器、不执行反向参数更新，只对选定病例计算预测类别 logit 对输入图像的梯度绝对值。每个模态只保存归一化后的中心/边缘比例和最多 `12` 张 PNG，不保存完整梯度张量。

命令行参数固定为：

```text
--run_dir
--case_audit_dir
--quality_flags
--output_dir
--device
--max_error_cases 24
--max_correct_cases 12
--max_maps 12
```

- [ ] **步骤 4：运行测试并提交代码**

```powershell
python -m pytest code/tests/test_model_evidence_audit.py -q
git add code/train/model_evidence_audit.py code/tests/test_model_evidence_audit.py
git commit -m "新增模型证据审计"
```

预期：纯函数测试全部通过，提交不包含模型权重和显著性张量。

- [ ] **步骤 5：远端执行模型证据审计**

```powershell
scp D:/Project/TinySpatial/code/train/model_evidence_audit.py anon-service2:/data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/train/model_evidence_audit.py
scp D:/Project/TinySpatial/code/tests/test_model_evidence_audit.py anon-service2:/data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/tests/test_model_evidence_audit.py
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python -m pytest code/tests/test_model_evidence_audit.py -q"
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/model_evidence_audit.py --run_dir /data/lzj813/b1-modality-ablation-20260809/full/flat5_20260809-112135 --case_audit_dir /data/lzj813/b1-case-audit-20260731-232334 --quality_flags /data/lzj813/b1-data-evidence-audit-20260811/data-quality/case_quality_flags.csv --output_dir /data/lzj813/b1-data-evidence-audit-20260811/model-evidence --device cuda:0 --max_error_cases 24 --max_correct_cases 12 --max_maps 12"
```

预期：生成 `model_evidence_summary.json`、`model_evidence_cases.csv`、`model_evidence_report.md` 和不超过 `12` 张可视化图。

## Task 3：结果汇总、方向判断和提交实验记录

**文件：**

- 创建：`docs/experiments/2026-08-11-data-quality-audit-results.md`
- 创建：`docs/experiments/2026-08-11-model-evidence-audit-results.md`

- [ ] **步骤 1：核对两类审计输出完整性**

```powershell
ssh anon-service2 "find /data/lzj813/b1-data-evidence-audit-20260811/data-quality -maxdepth 1 -type f -print"
ssh anon-service2 "find /data/lzj813/b1-data-evidence-audit-20260811/model-evidence -maxdepth 1 -type f -print"
```

预期：两个目录均存在 JSON、CSV 和中文 Markdown；模型证据图数量不超过 `12`。

- [ ] **步骤 2：按固定判定规则写入结论**

数据报告必须明确区分“风险标记”和“已确认错误”；模型报告必须明确区分“空间集中度”与“真实病灶定位”。若发现高优先级数据问题，结果文档将任务二的结论限制为确认性诊断，不追加模型训练。

- [ ] **步骤 3：提交两份脱敏实验记录**

```powershell
git add docs/experiments/2026-08-11-data-quality-audit-results.md docs/experiments/2026-08-11-model-evidence-audit-results.md
git -c user.name=LiZhijin-813 -c user.email=543521673@qq.com commit -m "记录数据质量与模型证据审计结果"
```

预期：提交只包含脱敏统计和中文结论，不包含原始文本、权重、缓存、日志或完整显著性张量；不执行 `git push`。
