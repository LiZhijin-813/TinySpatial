# B1 病例级错误审计设计

## 目标

在不改变 B1 的模型权重、数据切分、预处理和训练配置的前提下，对固定最优检查点的恶性测试集导出病例级预测，确认分子分型错误集中在哪些病例、真实亚型、预测亚型与置信度区间，并为后续人工核查标签和影像表现提供稳定清单。

本任务不进行重新训练，不引入 ROI，也不依据审计结果自动修改标签。审计输出用于发现需要人工复核的病例，不得被表述为标签纠正或性能提升。

## 范围与输入

新增独立脚本 `code/train/audit_stage2.py`，输入为一个完成的 `flat5` 运行目录。首轮唯一目标为 B1：`runs/stage2/flat5_20260731-232334`。

脚本必须从该目录读取 `args.json`、`split_manifest.json`、`best_model.pth` 和 `metrics_test.json`，并据此：

1. 恢复保存时的 `flat5` 模型配置；拒绝非 `flat5`、缺少必要文件或损坏的运行目录。
2. 使用 manifest 中的 `malignant_test` case ID 恢复病例，使用确定性评估变换与 `augment=False` 数据集。
3. 仅在推理模式下执行前向计算，不写入或覆盖原运行目录中的任何文件。
4. 输出到调用方指定的新目录；默认目录为运行目录下的 `case_audit`。若目录已存在且非空，必须显式指定 `--overwrite` 才能覆盖。

## 病例级输出

### `case_predictions.csv`

每个 `malignant_test` 病例恰有一行，按 manifest 顺序输出，包含：

- `case_id`。
- 真实亚型标签和名称：`true_subtype_label`、`true_subtype_name`。
- 条件四分类预测标签和名称：`predicted_subtype_label`、`predicted_subtype_name`。预测严格使用前四个 flat5 logits 的 argmax，与现有恶性条件 Macro-F1 语义一致。
- 端到端五分类预测标签和名称：`predicted_class_label`、`predicted_class_name`，用于识别被模型预测为良性的恶性病例。
- 完整五类 softmax 概率：`prob_luminal_a`、`prob_luminal_b`、`prob_her2`、`prob_tnbc`、`prob_benign`，其逐行和必须为 1。
- 条件四分类概率与置信度：由前四个 logits 单独 softmax 得到 `conditional_prob_luminal_a` 至 `conditional_prob_tnbc` 和 `conditional_confidence`；四类概率逐行和必须为 1。
- `is_conditional_correct`、`is_end_to_end_correct`、`error_type`。`error_type` 仅取 `正确`、`亚型错分`、`恶性病例预测为良性`。
- 模态可用性布尔值：`bus_exists`、`swe_exists`、`cdfi_exists`、`text_exists`。不导出原始图像、原始文本或患者身份信息。

### `case_audit_summary.json`

包含运行目录、检查点、测试病例数、各真实/预测亚型计数、错误类型计数、条件四分类指标和端到端五分类指标。条件四分类的 Accuracy、Balanced Accuracy、Macro-F1、混淆矩阵及预测分布必须与保存的 `metrics_test.json` 中恶性条件指标一致，浮点比较容差为 `1e-12`；不一致时脚本失败并给出中文错误。

### `error_audit.md`

面向人工复核的中文摘要，包含：

1. 审计来源和样本数。
2. 各真实亚型的总数、正确数、错误数及条件召回率。
3. 条件混淆矩阵和错误类型计数。
4. 所有错误病例，按条件置信度降序列出病例 ID、真实/预测亚型、端到端预测、条件置信度和良性概率。
5. 高置信度错分小节，阈值固定为 `0.80`；若没有病例，明确写出无高置信度错分。
6. 解释限制：该清单仅用于人工核查，不能单独证明标签错误或某种模态造成错误。

## 命令行与失败处理

脚本提供 `--run_dir`、`--output_dir`、`--device`、`--batch_size`、`--overwrite`。默认读取 CPU/GPU 均可，首轮远端使用 `cuda:0`。

对下列情况必须失败并给出中文错误：运行目录不完整、任务模式非 flat5、检查点加载失败、manifest 与恢复的病例不一致、病例数重复或缺失、概率非有限或不归一、输出指标无法复现保存指标、输出目录已存在且未请求覆盖。

## 测试与验收

新增专门的单元测试，使用最小伪数据和伪模型验证：

1. CSV 的列、行数、顺序、标签映射、概率归一和三种错误类型。
2. 汇总指标与已有评估函数一致，且当保存指标不一致时拒绝。
3. 非 flat5、缺失输入文件、重复/缺失 case ID、非空输出目录和非法概率均被拒绝并输出中文错误。
4. 审计过程不改变 `args.json`、`split_manifest.json`、`best_model.pth` 或 `metrics_test.json` 的内容。

验收时先运行新增目标测试与全量测试；随后在远端 B1 检查点运行一次审计。审计成功的必要条件是导出 82 个恶性测试病例、CSV 概率与汇总均可复现既有 B1 测试指标。

## GitHub 推送记录

2026-08-01 曾两次尝试非强制推送 `exp/reliable-stage2-baselines` 到 GitHub：第一次连接被重置，第二次无法连接 `github.com:443`。本地分支与远端服务器仓库均已保留至 `78152e3`；网络恢复后仅重试常规 `git push origin exp/reliable-stage2-baselines`。
