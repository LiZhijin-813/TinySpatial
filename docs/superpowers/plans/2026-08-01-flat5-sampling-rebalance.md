# B1 平衡采样替代重平衡机制执行计划

> **面向代理执行者：** 必须使用 `superpowers:subagent-driven-development` 按任务逐项执行，并维护复选框状态。

**目标：** 新增 `flat5` 交叉熵权重策略开关，并运行“普通交叉熵 + 平衡采样”的单种子可审计对照。

**架构：** 默认配置保持逆频率加权交叉熵。新增的 `flat5_class_weighting` 只在五分类任务中进入 `build_criteria_for_mode`，从而替换 B1 的重平衡机制而不影响其他任务。

**技术栈：** Python、argparse、PyTorch、pytest、SSH、远端 `tinyusfm` 环境。

## 全局约束

- 所有 Markdown、注释和运行日志使用中文。
- 默认 `flat5_class_weighting=inverse` 必须保持既有 B1 行为。
- `flat5_class_weighting=none` 只允许 `task_mode=flat5`。
- 新实验仅设置 `flat5_class_weighting=none` 与 `sampler=balanced`，其他 B1 参数不变。
- 模型选择只使用恶性验证集 `malignant_macro_f1`，测试集不参与配置选择。

---

### 任务 1：实现并测试 flat5 权重策略开关

**文件：**
- 修改：`code/train/train_stage2.py`
- 修改：`code/tests/test_run_artifacts.py`

**接口：**
- 消费：`build_parser()`、`validate_reliable_configuration(args)` 与 `build_criteria_for_mode(task_mode, samples, device, label_smoothing=0.0)`。
- 产出：`build_criteria_for_mode(..., flat5_class_weighting="inverse")` 和 `build_criteria_for_mode(..., flat5_class_weighting="none")`；主训练流程将 `args.flat5_class_weighting` 传入该函数。

- [ ] **步骤 1：写入失败测试**

在 `code/tests/test_run_artifacts.py` 增加三项测试：

```python
def test_flat5_cli_defaults_to_inverse_class_weighting():
    args = build_parser().parse_args(["--pretrained_path", "TinyUSFM.pth"])
    assert args.flat5_class_weighting == "inverse"

def test_flat5_unweighted_criterion_has_no_class_weights():
    criteria = build_criteria_for_mode(
        "flat5", _criterion_samples(), torch.device("cpu"),
        flat5_class_weighting="none",
    )
    assert criteria["class"].weight is None

def test_non_flat5_rejects_flat5_unweighted_strategy():
    args = build_parser().parse_args([
        "--pretrained_path", "TinyUSFM.pth", "--task_mode", "flat4",
        "--flat5_class_weighting", "none",
    ])
    with pytest.raises(ValueError):
        validate_reliable_configuration(args)
```

- [ ] **步骤 2：确认测试因功能缺失而失败**

运行：

```powershell
python -m pytest code/tests/test_run_artifacts.py -k "flat5_class_weighting or unweighted_criterion or non_flat5_rejects" -q
```

预期：测试因 `flat5_class_weighting` CLI 参数或函数参数尚不存在而失败，而非因导入或语法错误失败。

- [ ] **步骤 3：最小化实现**

在 `build_parser()` 中新增：

```python
parser.add_argument(
    "--flat5_class_weighting",
    choices=["inverse", "none"],
    default="inverse",
    help="flat5 五分类交叉熵类别权重策略",
)
```

将 `build_criteria_for_mode` 扩展为：

```python
def build_criteria_for_mode(
    task_mode,
    samples,
    device,
    label_smoothing=0.0,
    flat5_class_weighting="inverse",
):
```

在 `flat5` 分支中，仅当策略为 `inverse` 时计算 `compute_class_weights`；策略为 `none` 时把 `weight=None` 传给 `nn.CrossEntropyLoss`。未知策略抛出中文 `ValueError`。在 `validate_reliable_configuration` 中拒绝非 `flat5` 任务的 `none` 策略。主训练调用传入 `args.flat5_class_weighting`。

- [ ] **步骤 4：运行覆盖测试与完整测试套件**

运行：

```powershell
python -m pytest code/tests/test_run_artifacts.py -q
python -m pytest code/tests -q
```

预期：两条命令均以零失败结束。

- [ ] **步骤 5：提交代码与测试**

```powershell
git add code/train/train_stage2.py code/tests/test_run_artifacts.py
git commit -m "feat: 支持 flat5 平衡采样对照"
```

### 任务 2：运行并审计单种子对照

**文件：**
- 读取：远端 `runs/stage2/flat5_20260731-232334/`
- 创建：远端启动时间之后生成的 `runs/stage2/flat5_YYYYMMDD-HHMMSS/`
- 修改：`docs/experiments/2026-07-31-normalized-coordinate-baseline-results.md`

**接口：**
- 消费：任务 1 的 CLI 参数和现有固定切分。
- 产出：单种子运行产物与“进入三种子确认”或“拒绝”的记录。

- [ ] **步骤 1：同步已提交代码并启动远端实验**

运行：

```powershell
ssh anon-service2 "/bin/sh -lc 'cd /home/lzj813/TinySpatial_Project && touch /tmp/flat5_sampling_rebalance_start && CUDA_VISIBLE_DEVICES=0 /home/lzj813/miniconda3/envs/tinyusfm/bin/python -u code/train/train_stage2.py --task_mode flat5 --pretrained_path TinyUSFM.pth --epochs 100 --batch_size 8 --lr 5e-4 --wd 0.05 --unfreeze 2 --monitor_metric malignant_macro_f1 --seed 42 --sampler balanced --flat5_class_weighting none --device cuda:0'"
```

预期：运行目录的 `args.json` 显示 `sampler=balanced` 和 `flat5_class_weighting=none`，且不发生尺寸不一致错误。

- [ ] **步骤 2：核验产物、参数和切分**

确认启动时间哨兵之后的新 `flat5_YYYYMMDD-HHMMSS` 目录包含 `args.json`、`split_manifest.json`、`history.json`、`metrics_best.json`、`metrics_test.json`。逐字段比较其 `split_manifest.json` 与 B1 基线，所有 `splits` 数组必须相同。

- [ ] **步骤 3：根据预注册验证门槛决定后续运行**

读取 `metrics_best.json` 中的 `score`、恶性条件 `macro_f1` 和四类验证召回。仅在 `score >= 0.2645` 且四个召回均大于 0 时，使用种子 3407 与 2026 启动同配置确认；否则记录拒绝结论并停止该配置。

- [ ] **步骤 4：提交结果记录**

```powershell
git add docs/experiments/2026-07-31-normalized-coordinate-baseline-results.md
git commit -m "docs: 记录平衡采样替代重平衡结果"
```

## 自检

- 计划将“采样与加权交叉熵叠加”的问题替换为两个单一重平衡机制的比较。
- 代码修改范围限于 `flat5` 权重策略、参数校验和主训练调用；其他任务默认行为不变。
- 测试先于生产代码，并覆盖默认兼容、无权重准则和任务模式约束。
