# B1 平衡采样替代重平衡机制设计

## 目标

为 `flat5` 任务增加显式的五分类交叉熵权重策略，使“普通交叉熵 + 平衡采样”能够作为 B1“逆频率加权交叉熵 + 普通随机采样”的干净对照。该工作用于检验重平衡机制，而不是把采样和损失重加权叠加。

## 问题与依据

当前 `flat5` 默认使用按 `class_label` 逆频率计算的交叉熵权重；`--sampler balanced` 也按同一标签逆频率有放回采样。二者同时启用会改变少数类的样本暴露频率和单次损失权重，不能解释为单纯的采样效果。

## 接口设计

在 `code/train/train_stage2.py` 增加 CLI 参数：

- `--flat5_class_weighting`，取值为 `inverse` 或 `none`，默认 `inverse`。
- 参数只改变 `task_mode=flat5` 的 `nn.CrossEntropyLoss(weight=...)`。
- `inverse` 保持现有 `compute_class_weights(..., "class_label", 5)` 行为。
- `none` 构造 `nn.CrossEntropyLoss(weight=None, label_smoothing=...)`。
- 当任务模式不是 `flat5` 且显式设置 `flat5_class_weighting=none` 时，可靠配置校验必须以中文 `ValueError` 拒绝，避免出现无效且难审计的参数。

`args.json` 已自动保存所有 CLI 参数，因此该字段会随运行产物保留。`split_manifest.json`、模型结构、优化器、图像变换、验证加载器和测试加载器均不变。

## 实验定义

保留已完成 B1：逆频率加权交叉熵 + `sampler=none`。新实验固定全部 B1 参数，仅设置：

```text
task_mode=flat5
flat5_class_weighting=none
sampler=balanced
```

该新对照使用每个样本等权的交叉熵，并以 `WeightedRandomSampler` 平衡五分类的训练暴露频率。恶性验证 `malignant_macro_f1` 仍是唯一的模型选择指标。

## 测试与审计

1. CLI 默认值必须为 `inverse`，并接受 `none`。
2. `flat5_class_weighting=none` 时，五分类准则的 `weight` 必须为 `None`；默认策略仍保留长度为 5 的逆频率权重。
3. 非 `flat5` 任务设置 `none` 必须被可靠配置校验拒绝。
4. 单种子实验完成后，检查 `args.json`、所有切分数组和五个标准训练产物；B1 与新对照的切分必须一致。
5. 使用预注册的验证门槛决定是否展开种子 3407 与 2026：最佳恶性验证 Macro-F1 至少为 0.2645，且四个亚型验证召回均大于 0。

## 非目标

- 不改动 `flat4`、`dual_head` 或 `overfit` 的损失策略。
- 不同时引入焦点损失、标签平滑、对比损失、架构改动或新的数据增强。
- 不把单种子测试集结果作为配置选择依据。
