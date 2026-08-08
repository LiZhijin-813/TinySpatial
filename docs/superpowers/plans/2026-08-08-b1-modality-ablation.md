# B1 两阶段模态消融实施计划

> 对执行代理的要求：按任务逐项执行；每个任务先写失败测试，再写最小实现并独立验证。完成后提交代码和文档，但不提交模型权重、数据或 /data 实验产物。

**目标：** 为固定 B1 建立统一的模态屏蔽接口，完成固定检查点依赖诊断和独立训练消融，判断哪些模态组合能够改善恶性四分类。

**架构：** 在 MultiModalBreastDataset 的归一化后输入处实现统一模态屏蔽；训练入口和病例审计入口通过同一个命令行参数传递屏蔽集合。阶段 A 复用固定 B1 检查点和病例审计链路，阶段 B 复用现有 flat5 训练链路，只改变屏蔽集合并将输出根目录指向 /data。

**技术栈：** Python、PyTorch、pytest、现有 MultiModalBreastDataset、train_stage2.py、audit_stage2.py、远端 tinyusfm 环境和 RTX 4090。

## 全局约束

- 固定任务：flat5，固定 B1 切分清单和 seed=42。
- 主要指标：恶性条件 Macro-F1、四类 Recall、Balanced Accuracy、混淆矩阵和预测分布。
- 二分类 AUC 只能作为辅助指标，不能替代分型指标。
- 阶段 A 屏蔽结果只能用于模型依赖性诊断，不作为正式单模态性能结论。
- 图像屏蔽使用归一化后的零值张量；文本屏蔽使用全零 input_ids 和全零 attention_mask。
- 临时缓存、模型权重和实验输出统一写入远端 /data。
- 不自动修改标签，不修改原始 B1 运行目录，不提交数据和模型权重。

---

### 任务 1：实现统一模态屏蔽接口

**文件：**

- 修改：code/datasets/dataset.py
- 测试：code/tests/test_dataset.py

**接口：**

- 新增 VALID_ABLATION_MODALITIES = ("bus", "swe", "cdfi", "text")。
- 新增 normalize_ablate_modalities(values) -> frozenset[str]，拒绝未知模态，去除重复值；None 或空序列返回空集合。
- MultiModalBreastDataset.__init__ 新增 ablate_modalities=None 参数，并保存为规范化后的 frozenset。
- __getitem__ 在图像归一化和文本 tokenization 完成后，根据集合将被屏蔽输入替换为同形状零张量。

**步骤：**

- [ ] 步骤 1：先写失败测试

在 code/tests/test_dataset.py 增加以下测试：

~~~python
def test_dataset_ablation_zeroes_selected_modalities(
    tiny_multimodal_root,
    fake_tokenizer,
):
    root, samples = tiny_multimodal_root
    full = MultiModalBreastDataset(
        str(root),
        split="val",
        samples=[samples[0]],
        augment=False,
        tokenizer=fake_tokenizer,
    )[0]
    ablated = MultiModalBreastDataset(
        str(root),
        split="val",
        samples=[samples[0]],
        augment=False,
        tokenizer=fake_tokenizer,
        ablate_modalities=["swe", "text"],
    )[0]

    assert torch.count_nonzero(ablated["bus_img"]) > 0
    assert torch.count_nonzero(ablated["cdfi_img"]) > 0
    assert torch.count_nonzero(ablated["swe_img"]) == 0
    assert torch.count_nonzero(ablated["input_ids"]) == 0
    assert torch.count_nonzero(ablated["attention_mask"]) == 0
    assert torch.count_nonzero(full["swe_img"]) > 0


@pytest.mark.parametrize("values", [["unknown"], ["bus", "unknown"]])
def test_dataset_ablation_rejects_unknown_modalities(values):
    from code.datasets.dataset import normalize_ablate_modalities

    with pytest.raises(ValueError, match="模态"):
        normalize_ablate_modalities(values)
~~~

- [ ] 步骤 2：运行失败测试

运行：

~~~powershell
& 'D:\Program\Anocanda\envs\yolov8\python.exe' -m pytest code/tests/test_dataset.py -k 'ablation' -q
~~~

预期：失败，因为数据集构造函数尚未接受 ablate_modalities，且规范化函数尚不存在。

- [ ] 步骤 3：实现最小屏蔽逻辑

在 code/datasets/dataset.py 中实现：

~~~python
VALID_ABLATION_MODALITIES = ("bus", "swe", "cdfi", "text")


def normalize_ablate_modalities(values):
    values = () if values is None else tuple(values)
    unknown = set(values) - set(VALID_ABLATION_MODALITIES)
    if unknown:
        raise ValueError(f"未知模态：{sorted(unknown)}")
    return frozenset(values)
~~~

在 MultiModalBreastDataset.__init__ 保存 self.ablate_modalities；在 __getitem__ 中完成归一化和文本编码后加入：

~~~python
if "bus" in self.ablate_modalities:
    bus_tensor = torch.zeros_like(bus_tensor)
if "swe" in self.ablate_modalities:
    swe_tensor = torch.zeros_like(swe_tensor)
if "cdfi" in self.ablate_modalities:
    cdfi_tensor = torch.zeros_like(cdfi_tensor)
if "text" in self.ablate_modalities:
    input_ids = torch.zeros_like(encoding["input_ids"].squeeze(0))
    attention_mask = torch.zeros_like(encoding["attention_mask"].squeeze(0))
else:
    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)
~~~

- [ ] 步骤 4：运行测试并提交

运行：

~~~powershell
& 'D:\Program\Anocanda\envs\yolov8\python.exe' -m pytest code/tests/test_dataset.py -k 'ablation' -q
~~~

预期：相关测试通过。提交：

~~~powershell
git add code/datasets/dataset.py code/tests/test_dataset.py
git commit -m "feat: 增加统一模态屏蔽接口"
~~~

### 任务 2：将屏蔽配置接入训练与审计入口

**文件：**

- 修改：code/train/train_stage2.py
- 修改：code/train/audit_stage2.py
- 修改：code/train/case_audit.py
- 测试：code/tests/test_run_artifacts.py
- 测试：code/tests/test_case_audit.py

**接口：**

- 两个命令入口新增 --ablate_modalities，可接受 bus、swe、cdfi、text 的零个或多个值。
- _multimodal_dataset(samples, split, args, augment=False) 将 args.ablate_modalities 传入数据集。
- 检查点复评路径也从已保存参数读取 ablate_modalities，保证保存后的实验可以复现。
- case_audit_summary.json 新增 ablate_modalities 字段，保存排序后的屏蔽模态列表。

**步骤：**

- [ ] 步骤 1：先写失败测试

在训练入口测试中增加：

~~~python
def test_cli_accepts_ablation_modalities():
    args = build_parser().parse_args([
        "--pretrained_path", "TinyUSFM.pth",
        "--task_mode", "flat5",
        "--ablate_modalities", "swe", "text",
    ])

    assert args.ablate_modalities == ["swe", "text"]
~~~

在病例审计测试中增加：

~~~python
def test_audit_parser_accepts_ablation_modalities():
    args = audit_stage2_module.build_parser().parse_args([
        "--run_dir", "运行目录",
        "--ablate_modalities", "bus", "cdfi",
    ])

    assert args.ablate_modalities == ["bus", "cdfi"]
~~~

同时让汇总测试断言 summary["ablate_modalities"] == []，并增加传入 ("swe", "text") 后返回排序列表的测试。

- [ ] 步骤 2：运行失败测试

运行：

~~~powershell
& 'D:\Program\Anocanda\envs\yolov8\python.exe' -m pytest code/tests/test_run_artifacts.py code/tests/test_case_audit.py -k 'ablation' -q
~~~

预期：失败，因为两个解析器尚未注册参数，汇总结果尚无屏蔽配置字段。

- [ ] 步骤 3：实现参数传播

两个解析器使用同一组 choices：

~~~python
parser.add_argument(
    "--ablate_modalities",
    nargs="*",
    choices=VALID_ABLATION_MODALITIES,
    default=[],
    help="实验时屏蔽的模态列表",
)
~~~

训练入口在 _multimodal_dataset 和 evaluate_checkpoint 创建数据集时传入 ablate_modalities=getattr(args, "ablate_modalities", [])。病例审计入口同样将命令行参数传给 MultiModalBreastDataset，并把规范化后的列表写入汇总。

- [ ] 步骤 4：运行相关测试并提交

运行：

~~~powershell
& 'D:\Program\Anocanda\envs\yolov8\python.exe' -m pytest code/tests/test_run_artifacts.py code/tests/test_case_audit.py -q
~~~

预期：所有相关测试通过。提交：

~~~powershell
git add code/train/train_stage2.py code/train/audit_stage2.py code/train/case_audit.py code/tests/test_run_artifacts.py code/tests/test_case_audit.py
git commit -m "feat: 接入模态消融训练与审计参数"
~~~

### 任务 3：本地集成验证和远端同步

**文件：**

- 验证：code/datasets/dataset.py
- 验证：code/train/train_stage2.py
- 验证：code/train/audit_stage2.py
- 验证：code/tests/test_dataset.py
- 验证：code/tests/test_run_artifacts.py
- 验证：code/tests/test_case_audit.py

**步骤：**

- [ ] 步骤 1：运行完整相关测试

~~~powershell
& 'D:\Program\Anocanda\envs\yolov8\python.exe' -m pytest code/tests/test_dataset.py code/tests/test_run_artifacts.py code/tests/test_case_audit.py -q
~~~

预期：测试全部通过；允许存在 pytest 缓存目录无写权限警告，但不允许测试失败。

- [ ] 步骤 2：验证两个命令入口帮助信息

~~~powershell
& 'D:\Program\Anocanda\envs\yolov8\python.exe' code/train/train_stage2.py --help
& 'D:\Program\Anocanda\envs\yolov8\python.exe' code/train/audit_stage2.py --help
~~~

预期：退出码为 0，并能看到 --ablate_modalities。

- [ ] 步骤 3：提交前检查

~~~powershell
git diff --check
git status --short
~~~

不得出现未提交的代码修改；.hf_cache/ 等本地缓存不得加入暂存区。

- [ ] 步骤 4：推送代码分支

~~~powershell
git push git@github.com:LiZhijin-813/TinySpatial.git exp/reliable-stage2-baselines
~~~

随后使用 git ls-remote 核对远端分支哈希与本地 HEAD 一致。

### 任务 4：阶段 A 固定检查点依赖诊断

**输入：**

- 检查点：/home/lzj813/TinySpatial_Project/runs/stage2/flat5_20260731-232334
- 代码：远端项目中的已推送分支代码
- 输出根目录：/data/lzj813/b1-modality-probe-20260808

**步骤：**

- [ ] 步骤 1：检查远端磁盘和 GPU

~~~bash
ssh anon-service2 "df -h / /data"
ssh anon-service2 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader"
~~~

如果 GPU 全部高利用率，不启动任务；如果出现显存不足，停止当前运行并记录日志。

- [ ] 步骤 2：运行完整四模态诊断

~~~bash
HF_HOME=/data/lzj813/hf-cache TRANSFORMERS_CACHE=/data/lzj813/hf-cache /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/audit_stage2.py --run_dir /home/lzj813/TinySpatial_Project/runs/stage2/flat5_20260731-232334 --device cuda:2 --output_dir /data/lzj813/b1-modality-probe-20260808/full --overwrite
~~~

- [ ] 步骤 3：依次运行四个去模态诊断

~~~bash
HF_HOME=/data/lzj813/hf-cache TRANSFORMERS_CACHE=/data/lzj813/hf-cache /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/audit_stage2.py --run_dir /home/lzj813/TinySpatial_Project/runs/stage2/flat5_20260731-232334 --device cuda:2 --output_dir /data/lzj813/b1-modality-probe-20260808/without-bus --ablate_modalities bus --overwrite
HF_HOME=/data/lzj813/hf-cache TRANSFORMERS_CACHE=/data/lzj813/hf-cache /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/audit_stage2.py --run_dir /home/lzj813/TinySpatial_Project/runs/stage2/flat5_20260731-232334 --device cuda:2 --output_dir /data/lzj813/b1-modality-probe-20260808/without-swe --ablate_modalities swe --overwrite
HF_HOME=/data/lzj813/hf-cache TRANSFORMERS_CACHE=/data/lzj813/hf-cache /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/audit_stage2.py --run_dir /home/lzj813/TinySpatial_Project/runs/stage2/flat5_20260731-232334 --device cuda:2 --output_dir /data/lzj813/b1-modality-probe-20260808/without-cdfi --ablate_modalities cdfi --overwrite
HF_HOME=/data/lzj813/hf-cache TRANSFORMERS_CACHE=/data/lzj813/hf-cache /home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/audit_stage2.py --run_dir /home/lzj813/TinySpatial_Project/runs/stage2/flat5_20260731-232334 --device cuda:2 --output_dir /data/lzj813/b1-modality-probe-20260808/without-text --ablate_modalities text --overwrite
~~~

- [ ] 步骤 4：比较阶段 A 结果

读取每个目录的 case_audit_summary.json，只比较 Accuracy、Macro-F1、Balanced Accuracy、四类 Recall、混淆矩阵和预测分布。将结果写入 docs/experiments/2026-08-08-b1-modality-probe-results.md，不把该阶段结果表述为正式单模态性能。

### 任务 5：阶段 B 独立训练消融

**输入：**

- 预训练权重：/home/lzj813/TinySpatial_Project/TinyUSFM.pth
- 输出根目录：/data/lzj813/b1-modality-ablation-20260808
- 设备：阶段 A 完成后选择空闲 GPU，优先 cuda:2

**步骤：**

- [ ] 步骤 1：先运行 BUS-only 小样本烟雾测试

使用 --ablate_modalities swe cdfi text，只确认数据集、模型前传、损失和验证指标均能运行；烟雾测试不得写入正式结果目录。

- [ ] 步骤 2：顺序运行五个固定组合

~~~bash
/home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/train_stage2.py --pretrained_path TinyUSFM.pth --task_mode flat5 --seed 42 --device cuda:2 --output_root /data/lzj813/b1-modality-ablation-20260808/bus-only --ablate_modalities swe cdfi text
/home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/train_stage2.py --pretrained_path TinyUSFM.pth --task_mode flat5 --seed 42 --device cuda:2 --output_root /data/lzj813/b1-modality-ablation-20260808/bus-swe --ablate_modalities cdfi text
/home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/train_stage2.py --pretrained_path TinyUSFM.pth --task_mode flat5 --seed 42 --device cuda:2 --output_root /data/lzj813/b1-modality-ablation-20260808/bus-cdfi --ablate_modalities swe text
/home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/train_stage2.py --pretrained_path TinyUSFM.pth --task_mode flat5 --seed 42 --device cuda:2 --output_root /data/lzj813/b1-modality-ablation-20260808/bus-text --ablate_modalities swe cdfi
/home/lzj813/miniconda3/envs/tinyusfm/bin/python code/train/train_stage2.py --pretrained_path TinyUSFM.pth --task_mode flat5 --seed 42 --device cuda:2 --output_root /data/lzj813/b1-modality-ablation-20260808/full
~~~

每个任务必须串行执行；任一任务 OOM、数据读取失败或指标文件缺失时停止后续训练并记录原因。

- [ ] 步骤 3：保存结果摘要

从每个实际生成的 metrics_test.json、metrics_best.json 和 args.json 提取恶性条件 Macro-F1、四类 Recall、预测分布和二分类 AUC，写入 docs/experiments/2026-08-08-b1-modality-ablation-results.md。

### 任务 6：晋级判断、论文记录和最终提交

**文件：**

- 创建：docs/experiments/2026-08-08-b1-modality-probe-results.md
- 创建：docs/experiments/2026-08-08-b1-modality-ablation-results.md
- 修改：docs/experiments/2026-08-08-b1-case-audit.md

**步骤：**

- [ ] 步骤 1：按主判据筛选候选组合

只有验证集恶性条件 Macro-F1 超过 B1 0.2329，且四类验证召回均非零，或明确改善 Luminal B/TNBC 且整体指标没有明显下降的组合，才进入三种子复验。

- [ ] 步骤 2：运行候选组合的 seed 3407 和 2026

只复制候选组合的完整训练命令，将 --seed 分别改为 3407 和 2026，输出根目录分别改为对应 seed 子目录；不对被拒绝组合追加多种子训练。

- [ ] 步骤 3：更新论文实验记录

记录数据契约审计、阶段 A 依赖诊断、阶段 B 独立训练结果、患者级分组限制和最终候选判定。明确区分“诊断性观察”和“正式性能结果”。

- [ ] 步骤 4：运行最终验证并提交

~~~powershell
& 'D:\Program\Anocanda\envs\yolov8\python.exe' -m pytest code/tests/test_dataset.py code/tests/test_run_artifacts.py code/tests/test_case_audit.py -q
git diff --check
git status --short
git push git@github.com:LiZhijin-813/TinySpatial.git exp/reliable-stage2-baselines
~~~

最终只提交代码、测试和中文实验文档；/data 下的模型权重、缓存、CSV 和日志不进入 Git。
