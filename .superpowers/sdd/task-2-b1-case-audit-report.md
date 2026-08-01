# B1 病例级错误审计任务 2 报告

## 修改文件

- 新建 `code/train/audit_stage2.py`
- 修改 `code/tests/test_case_audit.py`
- 新建 `.superpowers/sdd/task-2-b1-case-audit-report.md`

未修改训练脚本、任务 1 模块、任务规格或实验文档，未运行远端实验。

## 恢复流程

命令行入口仅接受运行目录、可选输出目录、设备、可选批次大小和显式覆盖开关。入口读取运行目录中的 `args.json`、`split_manifest.json`、`best_model.pth` 与 `metrics_test.json`；校验保存参数和清单均为 `flat5`、保存批次大小为正整数、恶性测试病例非空且不重复。

随后以保存参数构造模型，使用 `strict=True` 加载 `model_state_dict`，依据保存清单恢复 `malignant_test` 的病例顺序，并以 `augment=False` 创建确定性数据集。收集器只前传 BUS、SWE、CDFI、文本编号和注意力掩码，收集病例编号、亚型标签、五分类分数及四种模态文件存在性；若加载器顺序与恢复样本顺序不完全一致则拒绝继续。

收集结果直接调用任务 1 的 `build_case_records`、`build_audit_summary` 和 `write_audit_outputs`。条件四分类指标必须在 `1e-12` 容差内复现保存指标，失败会传播且不会写入三个审计产物。

## 只读保证

实现不写入原运行目录内的四个输入产物。默认只在 `run_dir/case_audit` 创建新审计目录；非空目录仅在 `--overwrite` 下允许继续。测试保存并逐字节比较四个输入产物，确认运行审计后保持不变。

## 测试与验证

已遵循测试先行，先修改 `code/tests/test_case_audit.py` 后运行：

```powershell
pytest code/tests/test_case_audit.py -q
```

首次和最终复跑均在测试收集阶段失败：本地 `D:\Program\python3.10\python.exe` 与 `D:\Program\Anocanda\python.exe` 均未安装 `torch`，`code/tests/conftest.py` 导入时抛出 `ModuleNotFoundError: No module named 'torch'`。因此 pytest 没有执行，也不声明测试通过。

以下静态验证通过：

```powershell
py -3.10 -c "import ast; from pathlib import Path; [ast.parse(path.read_text(encoding='utf-8')) for path in (Path('code/train/audit_stage2.py'), Path('code/tests/test_case_audit.py'))]"
git diff --check
```

新增测试覆盖命令行参数、四个必需文件缺失、非 `flat5`、重复恶性病例、无效批次、不可用 CUDA、加载器顺序不一致、保存参数构造模型、严格权重加载、确定性数据集、三个产物写入、非空目录拒绝、运行产物字节不变，以及任务 1 指标复现失败传播。

## 提交

实现与测试提交：`5a86f12`

提交说明：`feat: 添加 B1 病例错误审计命令`

## 自审结论与疑虑

自审确认公共接口齐全，输出由任务 1 的公开接口统一生成，且新模块未复制指标、CSV、JSON 或 Markdown 的领域逻辑。输入运行产物仅读取，审计输出不含原始图像或原始文本。

主要疑虑是本地环境缺少 `torch`，导致无法执行 pytest 以获得运行时证据；静态语法和差异检查不能替代完整测试。报告提交与实现提交分开，以便报告记录实现提交的确切哈希。
