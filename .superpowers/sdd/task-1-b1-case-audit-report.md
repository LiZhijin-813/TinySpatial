# B1 病例级错误审计任务 1 报告

## 修改文件

- `code/train/case_audit.py`
- `code/tests/test_case_audit.py`
- `.superpowers/sdd/task-1-b1-case-audit-report.md`

## 实现字段与校验

实现了 flat5 `[N, 5]` logits 的病例记录构造、条件四分类指标复现、输出目录校验，以及 CSV、JSON、Markdown 三类审计产物写入。

每条记录包含病例 ID、真实亚型及名称、条件预测及名称、端到端预测及名称、两组独立归一化概率、条件置信度、两种正确性标记、三类错误类型和四种模态存在性。输出不写入原始图像、文本或患者身份信息。

校验覆盖空或重复病例 ID、标签和 logits 长度或形状不一致、非有限 logits、模态存在性数量或类型错误、保存指标缺失或无法在 `1e-12` 容差内复现，以及未经显式覆盖的非空输出目录。

## 测试与验证

已按 TDD 先新增 `code/tests/test_case_audit.py`，首次运行：

```powershell
pytest -q code/tests/test_case_audit.py
```

测试在收集阶段失败，当前 Python 环境未安装 `torch`：`ModuleNotFoundError: No module named 'torch'`。因此 pytest 完整通过计数为 0，未能执行新增的 11 个测试。

实现后再次执行同一命令，结果相同，仍在收集阶段被缺失的 `torch` 阻断。另执行：

```powershell
py -3.10 -m py_compile code/train/case_audit.py code/tests/test_case_audit.py
git -c safe.directory='D:/Project/TinySpatial' diff --check
```

## 审阅 P1/P2 修复

P1 根因：公开 `write_audit_outputs` 之前自行创建输出目录，未调用 `validate_output_directory`，使直接调用能够在非空目录写入固定产物文件。现增加 `overwrite=False` 公开参数，并在写入前统一调用目录校验；默认拒绝非空目录，显式 `overwrite=True` 才允许写入。测试新增直接写入非空目录拒绝和显式覆盖允许两种行为。

P2 根因：新增异常文本曾直接展示内部英文键名。现将病例记录、保存指标、恶性病例指标、分类分数、亚型标签和模态存在性等异常用户提示全部改为中文表达；新增测试确认缺失保存指标异常不暴露英文内部字段名。

本机本次仍执行：

```powershell
pytest code/tests/test_case_audit.py -q
```

结果仍在测试收集阶段因缺少 `torch` 报 `ModuleNotFoundError`，因此未声明 pytest 通过。以下静态检查通过：

```powershell
py -3.10 -c "import ast; from pathlib import Path; [ast.parse(path.read_text(encoding='utf-8')) for path in (Path('code/train/case_audit.py'), Path('code/tests/test_case_audit.py'))]"
git -c safe.directory='D:/Project/TinySpatial' diff --check
```

`py_compile` 因工作区 `__pycache__` 写入权限不足未能完成；`git diff --check` 未报告空白错误。

## 提交

实现与测试提交：`f7a7f493ba9d78be657575a3a292053ff3958a13`

提交信息：`feat: 添加病例级错误审计模块`

## 自审结论与疑虑

代码仅新增任务指定的两个代码/测试文件，另新增本报告；未创建命令行入口，未运行远端实验，未改动训练脚本或其他文档。主要疑虑是本地解释器缺少 `torch`，导致无法完成 pytest 运行验证；此外，受限环境阻止 `py_compile` 写入缓存文件。

## 远端失败修复

远端执行 `pytest code/tests/test_case_audit.py -q` 的失败结果为 `3 failed, 9 passed`。

根因一：测试夹具第二例真实标签为 `1`（Luminal B），而前四个 logits 为 `[0.0, 3.0, 2.0, -1.0]`，条件四分类 argmax 同样为 `1`。该例实际上是正确预测，却错误地期望“亚型错分”。现将 logits 改为 `[3.0, 0.0, 2.0, -1.0]`，使条件预测为 `0`（Luminal A），真实覆盖错分语义。

根因二：`_assert_reproducible` 使用 `if not saved` 判断缺失，递归到 `malignant.per_class.TNBC.recall` 的合法浮点值 `0.0` 时被误判为空。现只将 `None` 或空映射视为缺失，数值零继续参与有限性与 `1e-12` 容差比较。

修复后本机执行：

```powershell
pytest code/tests/test_case_audit.py -q
```

仍在收集阶段因环境缺少 `torch` 报 `ModuleNotFoundError`，未执行测试，因此不声明 pytest 通过。以下检查通过：

```powershell
py -3.10 -c "import ast; from pathlib import Path; [ast.parse(path.read_text(encoding='utf-8')) for path in (Path('code/train/case_audit.py'), Path('code/tests/test_case_audit.py'))]"
git -c safe.directory='D:/Project/TinySpatial' diff --check
```
