# Task 1 层级探针报告

## 任务范围

- 目标测试文件：`code/tests/test_hierarchical_probe.py`
- 目标实现文件：`code/train/run_cv_hierarchical_probe.py`
- 本次附加写入：`.superpowers/sdd/task-1-report.md`

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
..                                                                       [100%]
============================== warnings summary ===============================
... PytestCacheWarning: cache could not write path D:\Project\TinySpatial\.pytest_cache\v\cache\nodeids: [Errno 13] Permission denied
... PytestCacheWarning: cache could not write path D:\Project\TinySpatial\.pytest_cache\v\cache\stepwise: [Errno 13] Permission denied
2 passed, 2 warnings in 0.02s
```

## 实际修改文件

- `code/tests/test_hierarchical_probe.py`
- `code/train/run_cv_hierarchical_probe.py`

## 提交哈希

- `0ff1a17b58badd3c52865871217cfa817a514f32`

## 自检结论

- 层级探针测试已在带有 `torch` 的项目环境中通过。
- 除 `.hf_cache/` 的既有缓存脏目录外，没有新增其他工作区变更。
- 本次提交保持在任务要求的两个目标文件范围内，未修改设计或计划文档。
