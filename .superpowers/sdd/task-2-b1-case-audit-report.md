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

初始环境的主要疑虑是本地默认解释器缺少 `torch`，导致无法执行 pytest 以获得运行时证据；静态语法和差异检查不能替代完整测试。报告提交与实现提交分开，以便报告记录实现提交的确切哈希。

## 本机真实测试修复

使用 `D:\Program\Anocanda\envs\yolov8\python.exe` 执行真实测试后，发现 `test_run_case_audit_propagates_task1_metric_reproduction_failure` 初始失败。根因是测试默认清单的恶性测试病例顺序为 `[病例-B, 病例-A]`，但该测试的 canonical 样本和 mock 批次只有 `病例-A`；恢复划分先因未知病例而失败，尚未到达任务 1 的指标复现校验。

修复仅修改 `code/tests/test_case_audit.py`：将 canonical 样本和 mock 批次统一为与清单完全一致的 `[病例-B, 病例-A]`，并保留故意不完整的保存指标。测试现在会经过清单恢复，进入任务 1 指标复现失败路径，并确认审计目录没有写入产物。

```powershell
D:\Program\Anocanda\envs\yolov8\python.exe -m pytest code/tests/test_case_audit.py -q
```

实际结果：`26 passed, 3 warnings in 2.73s`。三条警告均为 `.pytest_cache` 无写入权限导致的 pytest 缓存警告，不影响测试执行结果。

修复提交：`bdc2d92`，提交说明：`test: 修复病例审计指标复现测试`。

最新疑虑仅为仓库 `.pytest_cache` 的写入权限会产生缓存警告；病例审计测试本身已在指定解释器中完整通过。

## 直接脚本入口修复

直接执行 `code/train/audit_stage2.py` 时，解释器首先将 `code/train` 放入模块搜索路径，标准库 `code` 因而遮蔽项目的 `code` 包。项目包导入在参数解析前失败，错误为“`code` 不是包”。

修复在 `audit_stage2.py` 的项目包导入前定义项目根目录、将其插入模块搜索路径首位，并在已加载的同名模块不是包时移除该模块。实现沿用 `train_stage2.py` 与 `inference.py` 的既有引导模式；从训练模块导入的同名项目根目录常量已移除。

新增子进程回归测试直接执行审计脚本的 `--help`，不读取真实模型或数据，并断言项目 `code` 包可被正确解析。

```powershell
D:\Program\Anocanda\envs\yolov8\python.exe code/train/audit_stage2.py --help
D:\Program\Anocanda\envs\yolov8\python.exe -m pytest code/tests/test_case_audit.py -q
```

实际结果：直接脚本帮助命令退出码为 0；完整病例审计测试为 `27 passed, 3 warnings in 7.45s`。警告仍仅涉及 `.pytest_cache` 无写入权限。

修复提交：`3b06429`，提交说明：`fix: 修复病例审计脚本导入路径`。
