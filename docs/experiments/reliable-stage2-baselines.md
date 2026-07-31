# Stage 2 可靠基线实验记录

## 数据契约

- 恶性训练/验证/测试：534/151/82，固定来自 `metadata.csv`。
- 额外良性训练/验证/测试：191/41/42，固定来自 `metadata_5class.csv`。
- 文件名基础组审计只作警告，不据此改动划分。

## 记录规则

- 门禁记录：运行目录、Seed、子集 case ID、最终训练损失、训练 Accuracy、预测分布和是否通过。
- 公平基线记录：运行目录、恶性 Macro-F1、Balanced Accuracy、四类 Recall、预测分布、logits 方差和特征方差。
- B1/B2 额外记录条件分型与端到端分型的差异；B2 记录良恶性 AUC、Sensitivity 和 Specificity。
- 所有数值只从对应目录的 `args.json`、`split_manifest.json`、`history.json`、`metrics_best.json`、`metrics_test.json` 和 `overfit_gate.json` 读取。

## 过拟合门禁事实

- 首次 32 例运行：`runs/stage2/overfit_20260731-190151`，Seed 42，最终 Accuracy `0.96875`，loss `0.03012810816289857`，预测分布 `[8, 8, 8, 8]`，未通过。该失败用于发现并修复过拟合门禁被默认 early stopping 截断的问题。
- 32 例通过运行：`runs/stage2/overfit_20260731-190842`，Seed 42，最终 Accuracy `1.0`，loss `0.00010101348561875056`，预测分布 `[8, 8, 8, 8]`，通过；其 `split_manifest.json` 记录了每类 8 例的 case ID。
- 64 例通过运行：`runs/stage2/overfit_20260731-191348`，Seed 42，最终 Accuracy `1.0`，loss `0.000021692125073968782`，预测分布 `[16, 16, 16, 16]`，通过；其 `split_manifest.json` 记录了每类 16 例的 case ID。
- 结论：BUS-only 探针、标签映射、分类损失、优化器参数和门禁审计链路均已通过可学习性验证，可进入公平 B0/B1/B2 基线阶段。
