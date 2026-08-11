# Task 1 层级探针修复报告

## 任务范围

- 测试文件：`code/tests/test_hierarchical_probe.py`
- 实现文件：`code/train/run_cv_hierarchical_probe.py`
- 报告文件：`.superpowers/sdd/task-1-report.md`

## 本轮修复内容

1. 补全 `build_hierarchical_tasks()` 的契约测试，完整断言三个任务键及其 `positive_labels`、`negative_labels`、`display_names`。
2. 增加 `select_hierarchical_labels()` 的非法输入测试，覆盖负数、一维以外张量、非整数张量三类输入。

本轮没有新增生产逻辑；现有实现已经满足新增测试，因此保持实现不变。

## RED

命令：

```powershell
python -m pytest code/tests/test_hierarchical_probe.py -q
```

输出：

```text
D:\Program\python3.10\python.exe: No module named pytest
```

补充检查：

```powershell
pytest code/tests/test_hierarchical_probe.py -q
```

输出：

```text
ImportError while loading conftest 'D:\Project\TinySpatial\code\tests\conftest.py'.
ModuleNotFoundError: No module named 'torch'
```

## GREEN

命令：

```powershell
& 'D:\Program\Anocanda\envs\yolov8\python.exe' -m pytest code/tests/test_hierarchical_probe.py -q
```

输出：

```text
......                                                                   [100%]
============================== warnings summary ===============================
... PytestCacheWarning: cache could not write path D:\Project\TinySpatial\.pytest_cache\v\cache\nodeids: [Errno 13] Permission denied
... PytestCacheWarning: cache could not write path D:\Project\TinySpatial\.pytest_cache\v\cache\stepwise: [Errno 13] Permission denied
6 passed, 2 warnings in 0.02s
```

## 实际修改文件

- `code/tests/test_hierarchical_probe.py`
- `.superpowers/sdd/task-1-report.md`

## 提交哈希

- 首次完成提交：`5a1001a7902e40da7c1c28a11f373a3919ff950c`
- 本轮修复提交：待提交后更新

## 自检结论

- 新增测试覆盖了任务配置完整性和非法标签输入边界。
- `select_hierarchical_labels()` 已能正确拒绝负数、非一维、非整数标签。
- 运行结果稳定为 `6 passed`，仅有 pytest 缓存目录写入警告。
- 除 `.hf_cache/` 和 pytest 缓存外，没有新增无关工作区改动。
