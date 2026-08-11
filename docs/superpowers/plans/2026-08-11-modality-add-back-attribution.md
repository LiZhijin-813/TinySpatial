# SWE/CDFI 逐一加回模态归因实施计划

> **面向执行代理：** 必须按任务逐项执行；推荐使用 `superpowers:subagent-driven-development`，每个任务完成后进行独立审查。步骤使用复选框跟踪。

**目标：** 在同协议条件下判断 SWE 或 CDFI 是否稳定贡献 Luminal A/B 的区分信息，并在没有稳定证据时自动停止新增训练。

**架构：** 先对阶段 B 已有的同协议 `bus-text` 与 `full` 检查点运行病例组五折层级探针，作为可比性门禁。门禁通过后，沿用现有 `train_stage2.py` 和 `--ablate_modalities` 接口，只新增 `bus-text-swe` 与 `bus-text-cdfi` 两次 `seed=42` 独立训练，再对新增检查点运行同一探针并汇总结果。

**技术栈：** Python、PyTorch、pytest、现有训练入口、病例组五折探针、远端 `tinyusfm` 环境和 RTX 4090。

## 全局约束

- 所有新增文档、注释、提示和错误信息使用中文。
- 固定任务为 `flat5`，随机种子为 `42`，监控指标为 `malignant_macro_f1`。
- 新增训练使用 `unfreeze=2`、`patience=20`、`flat5_class_weighting=inverse`、交叉熵损失和同一预训练权重。
- 两个基线必须来自同协议运行：`seed=42`、`unfreeze=2`、`patience=20`、逆频率类别权重。
- 验证集用于门禁和候选判断；测试集只允许作为最终一次性审计，不参与选择。
- 所有缓存、检查点、日志和探针结果写入远端 `/data`，不写入 Git 仓库。
- 训练串行执行；遇到非零退出、显存不足、数据读取失败或指标文件缺失，立即停止后续训练并记录原因。
- 本轮最多新增两次深度训练和四次轻量层级探针；不增加其他模态组合、随机种子或结构变体。
- 不推送 GitHub。

## 文件与职责

- 验证：`D:/Project/TinySpatial/code/train/train_stage2.py`，复用现有模态屏蔽训练入口。
- 验证：`D:/Project/TinySpatial/code/train/run_cv_hierarchical_probe.py`，复用现有病例组五折层级探针。
- 验证：`D:/Project/TinySpatial/code/tests/test_hierarchical_probe.py`，确认探针协议仍为 `44 passed`。
- 远端实验输出：`/data/lzj813/b1-modality-attribution-20260811`。
- 创建：`D:/Project/TinySpatial/docs/experiments/2026-08-11-modality-add-back-attribution-results.md`，只记录真实生成的指标和停止/晋级判断。

本计划不预期修改训练代码；只有验证暴露出现有接口无法表达本实验配置时，才暂停并单独修订计划。

### Task 1：同协议基线可比性门禁

**文件：**

- 验证：`code/train/run_cv_hierarchical_probe.py`
- 验证：`code/tests/test_hierarchical_probe.py`
- 读取：`/data/lzj813/b1-modality-ablation-20260809/bus-text/flat5_20260809-110754/args.json`
- 读取：`/data/lzj813/b1-modality-ablation-20260809/full/flat5_20260809-112135/args.json`
- 创建远端目录：`/data/lzj813/b1-modality-attribution-20260811/baseline-probe-*`

**接口：** 输入两个同协议检查点和 `/data/lzj813/b1-modality-diagnosis-20260809/cv-manifest-group5-seed42.json`；输出两个 `hierarchical_probe_metrics.json`。

- [ ] **步骤 1：同步探针代码并运行定向测试**

```powershell
scp D:/Project/TinySpatial/code/train/run_cv_hierarchical_probe.py anon-service2:/data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/train/run_cv_hierarchical_probe.py
scp D:/Project/TinySpatial/code/tests/test_hierarchical_probe.py anon-service2:/data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/tests/test_hierarchical_probe.py
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python -m pytest code/tests/test_hierarchical_probe.py -q"
```

预期：`44 passed`。失败则停止本计划。

- [ ] **步骤 2：核对基线参数**

```powershell
@'
import json
paths = [
    "/data/lzj813/b1-modality-ablation-20260809/bus-text/flat5_20260809-110754/args.json",
    "/data/lzj813/b1-modality-ablation-20260809/full/flat5_20260809-112135/args.json",
]
keys = ("seed", "patience", "unfreeze", "flat5_class_weighting", "monitor_metric", "task_mode")
for path in paths:
    data = json.load(open(path, encoding="utf-8"))
    print(path, {key: data.get(key) for key in keys})
'@ | ssh anon-service2 "/home/lzj813/miniconda3/envs/tinyusfm/bin/python -"
```

预期：两份参数均为 `42`、`20`、`2`、`inverse`、`malignant_macro_f1`、`flat5`。不一致则停止。

- [ ] **步骤 3：运行两个基线探针**

```powershell
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/run_cv_hierarchical_probe.py --run_dir /data/lzj813/b1-modality-ablation-20260809/bus-text/flat5_20260809-110754 --cv_manifest /data/lzj813/b1-modality-diagnosis-20260809/cv-manifest-group5-seed42.json --output_dir /data/lzj813/b1-modality-attribution-20260811/baseline-probe-bus-text --device cuda:0 --batch_size 16"
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/run_cv_hierarchical_probe.py --run_dir /data/lzj813/b1-modality-ablation-20260809/full/flat5_20260809-112135 --cv_manifest /data/lzj813/b1-modality-diagnosis-20260809/cv-manifest-group5-seed42.json --output_dir /data/lzj813/b1-modality-attribution-20260811/baseline-probe-full --device cuda:0 --batch_size 16"
```

预期：两个命令退出码为 `0)，三个任务有效样本数分别为 `767)、`354)、`413)。

- [ ] **步骤 4：计算门禁**

```powershell
@'
import json
from statistics import mean
paths = {
    "bus-text": "/data/lzj813/b1-modality-attribution-20260811/baseline-probe-bus-text/hierarchical_probe_metrics.json",
    "full": "/data/lzj813/b1-modality-attribution-20260811/baseline-probe-full/hierarchical_probe_metrics.json",
}
scores = {}
for name, path in paths.items():
    data = json.load(open(path, encoding="utf-8"))
    folds = data["tasks"]["luminal_a_vs_luminal_b"]["methods"]["plain"]
    scores[name] = [fold["metrics"]["macro_f1"] for fold in folds]
    print(name, [round(value, 4) for value in scores[name]], round(mean(scores[name]), 4))
delta = mean(scores["full"]) - mean(scores["bus-text"])
direction_count = sum(new >= old for new, old in zip(scores["full"], scores["bus-text"]))
print("delta=", round(delta, 4), "direction_count=", direction_count)
print("门禁=", delta >= 0.03 and direction_count >= 4)
'@ | ssh anon-service2 "/home/lzj813/miniconda3/envs/tinyusfm/bin/python -"
```

门禁通过条件：`delta >= 0.03) 且 `direction_count >= 4)。不通过则跳过 Task 2 和 Task 3，直接在 Task 4 记录“同协议下未复现先前差异，停止新增训练”。

### Task 2：新增配置训练前检查

**文件：**

- 验证：`code/train/train_stage2.py`
- 验证：`code/tests/test_run_artifacts.py`
- 远端输出根目录：`/data/lzj813/b1-modality-attribution-20260811`

- [ ] **步骤 1：验证现有模态接口**

```powershell
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python -m pytest code/tests/test_run_artifacts.py -k 'ablation or cli_accepts' -q"
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/train_stage2.py --help | rg -- '--ablate_modalities'"
```

预期：参数测试通过，帮助信息包含 `--ablate_modalities)。失败则停止，不修改训练逻辑。

- [ ] **步骤 2：检查 GPU 和磁盘**

```powershell
ssh anon-service2 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader"
ssh anon-service2 "df -h /data"
```

选择当前可用 GPU；若没有空闲 GPU，只暂停，不修改代码。

### Task 3：顺序训练两个逐一加回配置并运行探针

**文件：**

- 创建远端训练目录：`/data/lzj813/b1-modality-attribution-20260811/bus-text-swe`、`/data/lzj813/b1-modality-attribution-20260811/bus-text-cdfi`
- 创建远端探针目录：`/data/lzj813/b1-modality-attribution-20260811/probe-bus-text-swe`、`/data/lzj813/b1-modality-attribution-20260811/probe-bus-text-cdfi`

**接口：** `--ablate_modalities cdfi` 表示保留 BUS、SWE、Text；`--ablate_modalities swe` 表示保留 BUS、CDFI、Text。

- [ ] **步骤 1：训练 `bus-text-swe`**

```powershell
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/train_stage2.py \
  --pretrained_path /home/lzj813/TinySpatial_Project/TinyUSFM.pth \
  --task_mode flat5 --seed 42 --device cuda:0 \
  --patience 20 --unfreeze 2 --flat5_class_weighting inverse \
  --monitor_metric malignant_macro_f1 \
  --output_root /data/lzj813/b1-modality-attribution-20260811/bus-text-swe \
  --ablate_modalities cdfi"
```

预期：退出码为 `0)，生成唯一的 `flat5_*` 子目录，`args.json` 保存 `ablate_modalities=["cdfi"]`。

- [ ] **步骤 2：训练 `bus-text-cdfi`**

```powershell
ssh anon-service2 "cd /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H && /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/train_stage2.py \
  --pretrained_path /home/lzj813/TinySpatial_Project/TinyUSFM.pth \
  --task_mode flat5 --seed 42 --device cuda:0 \
  --patience 20 --unfreeze 2 --flat5_class_weighting inverse \
  --monitor_metric malignant_macro_f1 \
  --output_root /data/lzj813/b1-modality-attribution-20260811/bus-text-cdfi \
  --ablate_modalities swe"
```

预期：退出码为 `0)，生成唯一的 `flat5_*` 子目录，`args.json` 保存 `ablate_modalities=["swe"]`。

- [ ] **步骤 3：确认训练产物**

```powershell
ssh anon-service2 "find /data/lzj813/b1-modality-attribution-20260811/bus-text-swe -maxdepth 2 -type f \( -name args.json -o -name metrics_best.json -o -name metrics_test.json \) -print"
ssh anon-service2 "find /data/lzj813/b1-modality-attribution-20260811/bus-text-cdfi -maxdepth 2 -type f \( -name args.json -o -name metrics_best.json -o -name metrics_test.json \) -print"
```

预期：每组均有参数、最佳验证指标、测试指标和最佳检查点；只用 `metrics_best.json` 进行候选判断。

- [ ] **步骤 4：对两个新增检查点运行探针**

使用远端命令自动读取每组唯一的 `flat5_*` 子目录：

```powershell
ssh anon-service2 "run_dir=\$(find /data/lzj813/b1-modality-attribution-20260811/bus-text-swe -mindepth 1 -maxdepth 1 -type d -name 'flat5_*' | sort | tail -n 1); /home/lzj813/miniconda3/envs/tinyusfm/bin/python /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/train/run_cv_hierarchical_probe.py \
  --run_dir \$run_dir \
  --cv_manifest /data/lzj813/b1-modality-diagnosis-20260809/cv-manifest-group5-seed42.json \
  --output_dir /data/lzj813/b1-modality-attribution-20260811/probe-bus-text-swe \
  --device cuda:0 --batch_size 16"

ssh anon-service2 "run_dir=\$(find /data/lzj813/b1-modality-attribution-20260811/bus-text-cdfi -mindepth 1 -maxdepth 1 -type d -name 'flat5_*' | sort | tail -n 1); /home/lzj813/miniconda3/envs/tinyusfm/bin/python /data/lzj813/TinySpatial_probe_8d5b22b_20260808_Ts4H/code/train/run_cv_hierarchical_probe.py \
  --run_dir \$run_dir \
  --cv_manifest /data/lzj813/b1-modality-diagnosis-20260809/cv-manifest-group5-seed42.json \
  --output_dir /data/lzj813/b1-modality-attribution-20260811/probe-bus-text-cdfi \
  --device cuda:0 --batch_size 16"
```

预期：两个命令退出码为 `0)，各生成 `hierarchical_probe_metrics.json`。

### Task 4：统一汇总、晋级判断和提交实验记录

**文件：**

- 创建：`docs/experiments/2026-08-11-modality-add-back-attribution-results.md`
- 不提交：`/data/lzj813/b1-modality-attribution-20260811` 下的权重、缓存、日志和中间特征。

- [ ] **步骤 1：从四组真实 JSON 汇总三个层级**

```powershell
@'
import json
from pathlib import Path
from statistics import mean, pstdev

paths = {
    "bus-text": Path("/data/lzj813/b1-modality-attribution-20260811/baseline-probe-bus-text/hierarchical_probe_metrics.json"),
    "full": Path("/data/lzj813/b1-modality-attribution-20260811/baseline-probe-full/hierarchical_probe_metrics.json"),
    "bus-text-swe": Path("/data/lzj813/b1-modality-attribution-20260811/probe-bus-text-swe/hierarchical_probe_metrics.json"),
    "bus-text-cdfi": Path("/data/lzj813/b1-modality-attribution-20260811/probe-bus-text-cdfi/hierarchical_probe_metrics.json"),
}
for name, path in paths.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    print(name)
    for task, value in data["tasks"].items():
        folds = value["methods"]["plain"]
        macro_f1 = [fold["metrics"]["macro_f1"] for fold in folds]
        balanced_accuracy = [fold["metrics"]["balanced_accuracy"] for fold in folds]
        print(task, round(mean(macro_f1), 4), round(pstdev(macro_f1), 4), round(mean(balanced_accuracy), 4))
'@ | ssh anon-service2 "/home/lzj813/miniconda3/envs/tinyusfm/bin/python -"
```

预期：只从真实 JSON 读取指标，不手工修改原始结果。

- [ ] **步骤 2：应用固定晋级规则**

新增配置必须同时满足：相对 `bus-text` 的 LA/LB 五折 Macro-F1 提升至少 `0.03)；至少四折不低于基线；验证集恶性四分类 Macro-F1 下降不超过 `0.01)；另外两个层级没有预测坍缩。两个配置都通过时，只选择 LA/LB 提升更大者复验；都不通过时停止，不追加训练。

- [ ] **步骤 3：创建并提交中文实验记录**

记录四个配置的实际模态、参数、运行路径、提交哈希、验证指标、三个层级五折指标、门禁结果、晋级判断和停止原因，并明确说明该实验不能单独证明模态因果贡献。

```powershell
git add docs/experiments/2026-08-11-modality-add-back-attribution-results.md
git -c user.name=LiZhijin-813 -c user.email=543521673@qq.com commit -m "记录逐一加回模态归因实验结果"
```

预期：提交只包含中文实验记录，不包含远端权重、缓存、日志或未跟踪审查报告；不执行 `git push`。
