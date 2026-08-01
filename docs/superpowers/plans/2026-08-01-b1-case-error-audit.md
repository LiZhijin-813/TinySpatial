# B1 病例级错误审计实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** 为固定 B1 \`flat5\` 最优检查点导出可复现的恶性测试集病例级预测与错误审计报告，辅助人工复核错误病例和标签。

**Architecture:** 新建 \`case_audit.py\` 作为无训练副作用的领域模块，负责记录、指标复现、CSV/JSON/Markdown 生成和输入输出校验。新建 \`audit_stage2.py\` 作为薄命令行入口，复用现有运行配置恢复、数据切分恢复、模型构造和确定性评估数据集；它只协调 I/O 与模型前向，不复制训练代码。

**Tech Stack:** Python 3、PyTorch、标准 \`csv\` 模块、JSON、现有 \`MultiModalBreastDataset\`、\`SECSubtypingModel\`、\`evaluate_predictions\`、pytest。

## 全局约束

- 所有新增 Markdown、命令行提示、日志、异常、注释和文档字符串均使用中文。
- 首轮仅支持 \`flat5\`，拒绝其他 \`task_mode\`；不重新训练、不改变模型权重、不修改任何已有运行产物。
- 必须使用保存的 \`args.json\`、\`split_manifest.json\`、\`best_model.pth\`、\`metrics_test.json\`；恶性测试病例顺序严格等于 manifest 的 \`malignant_test\`。
- 全部输出写入调用方指定的新审计目录；非空目录只有显式 \`--overwrite\` 才能覆盖。
- 输出不包含原始图像、文本或患者身份信息；仅保留项目内 case ID 与模态文件是否存在。
- 条件四分类预测来自前四个 flat5 logits 的 argmax；五类概率和条件四类概率必须分别按行归一。
- 条件四分类指标必须在 \`1e-12\` 容差内复现保存的 B1 \`metrics_test.json\` 恶性指标。

---

### 任务 1：病例审计领域模块与可序列化产物

**Files:**
- Create: \`code/train/case_audit.py\`
- Test: \`code/tests/test_case_audit.py\`

**Interfaces:**
- Consumes: \`torch.Tensor\` 的 \`[N, 5]\` flat5 logits、按 DataLoader 顺序的 case ID/亚型标签、\`code.utils.evaluation.evaluate_predictions\`。
- Produces: \`build_case_records(case_ids, subtype_labels, class_logits, modality_exists) -> list[dict]\`、\`build_audit_summary(records, saved_metrics) -> dict\`、\`write_audit_outputs(output_dir, records, summary) -> None\`、\`validate_output_directory(output_dir, overwrite) -> Path\`。

- [ ] **步骤 1：写出病例记录与错误类型的失败测试**

在 \`code/tests/test_case_audit.py\` 创建最小 logits，覆盖正确、条件亚型错分和端到端预测为良性三种情形：

\`\`\`python
def test_build_case_records_keeps_order_probabilities_and_error_types():
    logits = torch.tensor([
        [5.0, 1.0, 0.0, 0.0, -1.0],
        [0.0, 4.0, 1.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, 0.0, 6.0],
    ])
    records = build_case_records(
        ["A", "B", "C"], [0, 2, 3], logits,
        [{"bus": True, "swe": True, "cdfi": True, "text": True}] * 3,
    )
    assert [row["case_id"] for row in records] == ["A", "B", "C"]
    assert [row["error_type"] for row in records] == ["正确", "亚型错分", "恶性病例预测为良性"]
    assert all(abs(sum(row[f"prob_{name}"] for name in PROBABILITY_NAMES) - 1.0) < 1e-12 for row in records)
    assert all(abs(sum(row[f"conditional_prob_{name}"] for name in SUBTYPE_PROBABILITY_NAMES) - 1.0) < 1e-12 for row in records)
\`\`\`

- [ ] **步骤 2：运行测试，确认模块尚未定义而失败**

运行：\`pytest code/tests/test_case_audit.py::test_build_case_records_keeps_order_probabilities_and_error_types -q\`

预期：失败，提示无法导入 \`code.train.case_audit\` 或 \`build_case_records\`。

- [ ] **步骤 3：实现最小记录构造与概率校验**

在 \`code/train/case_audit.py\` 定义：

\`\`\`python
SUBTYPE_NAMES = ("Luminal A", "Luminal B", "HER2+", "TNBC")
FIVE_CLASS_NAMES = SUBTYPE_NAMES + ("Benign",)

def build_case_records(case_ids, subtype_labels, class_logits, modality_exists):
    _validate_flat5_inputs(case_ids, subtype_labels, class_logits, modality_exists)
    class_probabilities = class_logits.softmax(dim=1)
    conditional_probabilities = class_logits[:, :4].softmax(dim=1)
    conditional_predictions = conditional_probabilities.argmax(dim=1)
    class_predictions = class_probabilities.argmax(dim=1)
    # 对每例写入既定 CSV 字段，并以端到端良性优先标记错误类型。
\`\`\`

\`_validate_flat5_inputs\` 必须拒绝空批、重复 case ID、非 \`[N, 5]\` logits、长度不一致、非有限 logits/概率和模态存在性长度不一致，并给出中文 \`ValueError\`。所有数值概率转换为有限 float。

- [ ] **步骤 4：扩展摘要、指标复现和目录校验的失败测试**

新增：

\`\`\`python
def test_build_audit_summary_rejects_saved_metric_mismatch():
    with pytest.raises(ValueError, match="无法复现"):
        build_audit_summary(_three_records(), {"malignant": {"macro_f1": 0.9}})

def test_validate_output_directory_rejects_nonempty_directory(tmp_path):
    output_dir = tmp_path / "case_audit"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("旧结果", encoding="utf-8")
    with pytest.raises(ValueError, match="非空"):
        validate_output_directory(output_dir, overwrite=False)
\`\`\`

- [ ] **步骤 5：实现摘要、三种文件写入和输出目录保护**

实现：

\`\`\`python
def build_audit_summary(records, saved_metrics):
    labels = np.array([row["true_subtype_label"] for row in records])
    predictions = np.array([row["predicted_subtype_label"] for row in records])
    conditional_metrics = evaluate_predictions(labels, predictions, SUBTYPE_NAMES)
    _assert_metrics_match(conditional_metrics, saved_metrics["malignant"])
    return {
        "sample_count": len(records),
        "conditional_metrics": conditional_metrics,
        "error_type_counts": _count_error_types(records),
        "per_true_subtype": _summarize_true_subtypes(records),
    }
\`\`\`

\`write_audit_outputs\` 用 \`csv.DictWriter\` 写 \`case_predictions.csv\`，用项目既有 \`save_json\` 写 \`case_audit_summary.json\`，并写中文 \`error_audit.md\`。Markdown 必须含来源、各亚型正确/错误/召回、混淆矩阵、全部错误病例、阈值 \`0.80\` 的高置信度错分和解释限制。

- [ ] **步骤 6：运行领域模块测试并提交**

运行：\`pytest code/tests/test_case_audit.py -q\`

预期：通过，覆盖行顺序、字段、概率归一、中文错误、指标不一致拒绝和非空目录保护。

提交：

\`\`\`powershell
git add code/train/case_audit.py code/tests/test_case_audit.py
git commit -m "feat: 添加病例级错误审计模块"
\`\`\`

### 任务 2：只读 B1 运行恢复与审计命令行入口

**Files:**
- Create: \`code/train/audit_stage2.py\`
- Modify: \`code/tests/test_case_audit.py\`

**Interfaces:**
- Consumes: 任务 1 的四个公开接口，以及 \`train_stage2.build_model\`、\`train_stage2.build_fair_splits\`、\`train_stage2.restore_manifest_splits\`、\`train_stage2.build_eval_loader\`、\`MultiModalBreastDataset\`。
- Produces: \`build_parser() -> argparse.ArgumentParser\`、\`load_flat5_audit_run(run_dir) -> tuple[Namespace, dict, Path, dict]\`、\`run_case_audit(args) -> Path\`。

- [ ] **步骤 1：写出运行目录完整性、任务模式和不改写原产物的失败测试**

\`\`\`python
def test_run_case_audit_rejects_non_flat5_run(tmp_path):
    run_dir = _write_audit_run(tmp_path, task_mode="dual_head")
    args = build_parser().parse_args(["--run_dir", str(run_dir)])
    with pytest.raises(ValueError, match="flat5"):
        run_case_audit(args)

def test_run_case_audit_writes_new_outputs_without_changing_run_files(tmp_path, monkeypatch):
    run_dir = _write_audit_run(tmp_path, task_mode="flat5")
    before = {name: (run_dir / name).read_bytes() for name in REQUIRED_RUN_FILES}
    _patch_tiny_audit_dependencies(monkeypatch)
    output_dir = run_case_audit(build_parser().parse_args(["--run_dir", str(run_dir)]))
    assert (output_dir / "case_predictions.csv").is_file()
    assert before == {name: (run_dir / name).read_bytes() for name in REQUIRED_RUN_FILES}
\`\`\`

- [ ] **步骤 2：运行命令行测试，确认入口尚不存在而失败**

运行：\`pytest code/tests/test_case_audit.py -q\`

预期：失败，提示无法导入 \`code.train.audit_stage2\`。

- [ ] **步骤 3：实现参数解析与严格运行恢复**

\`\`\`python
def build_parser():
    parser = argparse.ArgumentParser(description="导出 Stage 2 病例级错误审计")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser

def load_flat5_audit_run(run_dir):
    # 检查四个必需文件，读取 JSON，要求 args/manifest 均为 flat5，
    # 并要求 manifest 含有非空且无重复的 malignant_test。
\`\`\`

读取 JSON、检查点加载、模型构造和恢复 split 的异常必须转换为包含原因的中文 \`ValueError\`。缺省 batch size 使用保存值且必须为正数；选择 \`cuda:*\` 而设备不可用时必须先给出中文错误。

- [ ] **步骤 4：实现前向收集和可追溯模态存在性**

\`\`\`python
@torch.no_grad()
def collect_flat5_audit_inputs(model, loader, device, dataset):
    # 保持 batch 内 case_id 顺序，收集 subtype_label、class_logits。
    # 使用 dataset 的 bus_dir/swe_dir/cdfi_dir/texts_dir 生成每例 exists 标志。
    # 最终 case_id 序列必须等于 dataset.samples 的 case_id 序列，否则失败。

def run_case_audit(args):
    saved_args, manifest, checkpoint_path, saved_metrics = load_flat5_audit_run(args.run_dir)
    # build_fair_splits 后 restore_manifest_splits，创建 augment=False 的 malignant_test 数据集。
    # 加载 state_dict，调用 collect_flat5_audit_inputs，再调用任务 1 的领域接口写出新目录。
\`\`\`

模型检查点必须以 \`strict=True\` 加载；审计完成后只打印中文输出目录和病例数，不在原运行目录写入文件。

- [ ] **步骤 5：补齐 manifest 顺序、缺失文件、输出覆盖和指标复现测试**

\`\`\`python
def test_run_case_audit_rejects_manifest_case_order_mismatch(tmp_path, monkeypatch):
    run_dir = _write_audit_run(tmp_path, task_mode="flat5")
    _patch_tiny_audit_dependencies(monkeypatch, returned_case_ids=["B", "A"])
    with pytest.raises(ValueError, match="顺序"):
        run_case_audit(build_parser().parse_args(["--run_dir", str(run_dir)]))

def test_run_case_audit_requires_overwrite_for_existing_output(tmp_path, monkeypatch):
    run_dir = _write_audit_run(tmp_path, task_mode="flat5")
    output_dir = run_dir / "case_audit"
    output_dir.mkdir()
    (output_dir / "old.csv").write_text("旧结果", encoding="utf-8")
    _patch_tiny_audit_dependencies(monkeypatch)
    with pytest.raises(ValueError, match="非空"):
        run_case_audit(build_parser().parse_args(["--run_dir", str(run_dir)]))
\`\`\`

- [ ] **步骤 6：运行目标测试与全量测试并提交**

运行：

\`\`\`powershell
pytest code/tests/test_case_audit.py -q
pytest code/tests -q
\`\`\`

预期：目标和全量测试均通过。

提交：

\`\`\`powershell
git add code/train/audit_stage2.py code/tests/test_case_audit.py
git commit -m "feat: 添加 B1 病例错误审计命令"
\`\`\`

### 任务 3：远端 B1 审计运行、人工核查清单与实验记录

**Files:**
- Modify: \`docs/experiments/2026-07-31-normalized-coordinate-baseline-results.md\`
- Create: \`docs/experiments/2026-08-01-b1-case-audit.md\`

**Interfaces:**
- Consumes: 任务 2 的 \`audit_stage2.py\`，远端 B1 \`runs/stage2/flat5_20260731-232334\`，生成的三个审计产物。
- Produces: 一份只含汇总与病例 ID 的可核查实验记录，不复制敏感模态数据。

- [ ] **步骤 1：在远端运行固定 B1 审计命令**

先通过 Git bundle 将已测试提交同步至 \`/home/lzj813/TinySpatial_Project\`，快进后执行：

\`\`\`sh
/home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/audit_stage2.py \\
  --run_dir runs/stage2/flat5_20260731-232334 \\
  --device cuda:0
\`\`\`

预期：创建新的 \`case_audit/\`，包含三个文件，输出病例数为 82；不得运行训练命令。

- [ ] **步骤 2：执行产物一致性检查**

\`\`\`sh
/home/lzj813/miniconda3/envs/tinyusfm/bin/python -c "import csv, json, pathlib; root=pathlib.Path('runs/stage2/flat5_20260731-232334/case_audit'); rows=list(csv.DictReader((root/'case_predictions.csv').open(encoding='utf-8'))); summary=json.loads((root/'case_audit_summary.json').read_text(encoding='utf-8')); assert len(rows)==82; assert summary['sample_count']==82; print('病例审计产物一致')"
\`\`\`

预期：输出 \`病例审计产物一致\`；再比对 \`summary["conditional_metrics"]\` 与 B1 的 \`metrics_test.json["malignant"]\`，确认关键指标在 \`1e-12\` 内一致。

- [ ] **步骤 3：写入不夸大结论的实验记录**

在 \`docs/experiments/2026-08-01-b1-case-audit.md\` 记录运行目录、代码提交、82 例审计成功、每真实亚型错误计数、高置信度错分病例 ID 和人工核查问题。明确审计只定位复核对象，不证明标签错误、不构成性能提升、也不构成任何模态的因果证据。

在 \`docs/experiments/2026-07-31-normalized-coordinate-baseline-results.md\` 增加一行摘要，说明 B1 后续诊断转为病例级错误核查。

- [ ] **步骤 4：复核文档、提交并进行代码审阅**

运行：

\`\`\`powershell
git diff --check
pytest code/tests -q
\`\`\`

预期：无 diff 格式错误，测试全通过。

提交：

\`\`\`powershell
git add docs/experiments/2026-07-31-normalized-coordinate-baseline-results.md docs/experiments/2026-08-01-b1-case-audit.md
git commit -m "docs: 记录 B1 病例错误审计结果"
\`\`\`

随后执行独立代码审阅，重点检查审计是否只读、CSV 是否不泄露原始文本/图像、条件分型语义是否与 B1 一致、指标复现检查是否真实有效、结论是否越过人工核查边界。

## 计划自检

- 规格覆盖：任务 1 实现病例表、JSON、Markdown、概率和目录保护；任务 2 实现只读恢复、flat5 限制、manifest 顺序和原运行产物不变；任务 3 执行远端 B1 审计、82 例验证与文档记录。
- 占位符：已检查，无 \`TODO\`、\`TBD\`、待定项或未定义接口。
- 类型一致性：任务 1 的四个公开接口均由任务 2 调用；任务 2 的 \`run_case_audit(args) -> Path\` 由任务 3 的命令行使用；记录字段名称与规格中 CSV 字段一致。

