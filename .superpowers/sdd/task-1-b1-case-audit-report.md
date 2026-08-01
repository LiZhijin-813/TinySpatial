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

`py_compile` 因工作区 `__pycache__` 写入权限不足未能完成；`git diff --check` 未报告空白错误。

## 提交

实现与测试提交：`f7a7f493ba9d78be657575a3a292053ff3958a13`

提交信息：`feat: 添加病例级错误审计模块`

## 自审结论与疑虑

代码仅新增任务指定的两个代码/测试文件，另新增本报告；未创建命令行入口，未运行远端实验，未改动训练脚本或其他文档。主要疑虑是本地解释器缺少 `torch`，导致无法完成 pytest 运行验证；此外，受限环境阻止 `py_compile` 写入缓存文件。
