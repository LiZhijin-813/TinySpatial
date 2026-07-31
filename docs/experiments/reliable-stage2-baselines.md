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
