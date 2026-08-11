# 2026-08-11 同协议逐一加回模态归因实验结果

## 实验目的

本轮原计划通过 `BUS+Text`、`BUS+Text+SWE`、`BUS+Text+CDFI` 和 `Full` 四组结果，判断 SWE 或 CDFI 是否主要贡献了 Luminal A/B 的区分信息。

执行前的协议复核发现，先前使用的两个层级探针检查点并非同协议：`BUS+Text` 使用 `unfreeze=0` 和无类别权重，全模态使用 `unfreeze=2`、逆频率类别权重和 `patience=50`。因此，先前的 LA/LB 差异不能直接归因于模态。

按照预设门禁，本轮先使用阶段 B 中同协议的已有检查点进行轻量层级探针。门禁未通过，因此没有启动两个新增深度训练。

## 固定协议

两个基线检查点均核对为：

- `seed=42`
- `patience=20`
- `unfreeze=2`
- `flat5_class_weighting=inverse`
- `monitor_metric=malignant_macro_f1`
- `task_mode=flat5`

探针使用同一病例组五折清单：

`/data/lzj813/b1-modality-diagnosis-20260809/cv-manifest-group5-seed42.json`

远端定向测试结果为 `44 passed`。

## 远端结果路径

- `BUS+Text`：`/data/lzj813/b1-modality-attribution-20260811/baseline-probe-bus-text/hierarchical_probe_metrics.json`
- `Full`：`/data/lzj813/b1-modality-attribution-20260811/baseline-probe-full/hierarchical_probe_metrics.json`

两个结果的有效样本数均为：

- `luminal_vs_non_luminal`：`767`
- `luminal_a_vs_luminal_b`：`354`
- `her2_vs_tnbc`：`413`

## 层级探针结果

| 配置 | Luminal/non-Luminal Macro-F1 | Luminal A/B Macro-F1 | HER2+/TNBC Macro-F1 |
| --- | ---: | ---: | ---: |
| `BUS+Text` | `0.5633 ± 0.0140` | `0.5405 ± 0.0271` | `0.5229 ± 0.0378` |
| `Full` | `0.5364 ± 0.0390` | `0.5691 ± 0.0252` | `0.4917 ± 0.0342` |

对应 Balanced Accuracy 均值如下：

| 配置 | Luminal/non-Luminal | Luminal A/B | HER2+/TNBC |
| --- | ---: | ---: | ---: |
| `BUS+Text` | `0.5670` | `0.5425` | `0.5363` |
| `Full` | `0.5377` | `0.5719` | `0.5134` |

Luminal A/B 五折 Macro-F1 逐折结果为：

- `BUS+Text`：`[0.5902, 0.5419, 0.5093, 0.5262, 0.5348]`
- `Full`：`[0.5754, 0.5492, 0.5337, 0.5815, 0.6056]`

`Full` 相对 `BUS+Text` 的 LA/LB Macro-F1 均值提升为 `0.0286`，其中 `4/5` 折不低于 `BUS+Text`。

## 门禁判断

预设门禁为：

1. LA/LB 五折平均 Macro-F1 提升至少 `0.03`；
2. 至少 `4/5` 折不低于基线。

当前第二项满足，第一项不满足：`0.0286 < 0.03`。因此门禁不通过。

## 当前结论

在严格同协议对照下，可以确认全模态表示对 LA/LB 仍有接近门槛的增量信息，但提升幅度不足以支持启动 SWE/CDFI 逐一加回训练，也不足以证明某个具体模态的独立贡献。

同时，`Full` 在另外两个层级并没有优势：Luminal/non-Luminal 下降，HER2+/TNBC 也下降。这与“额外模态主要改善 LA/LB”的工作线索一致，但该线索尚未达到单模态归因实验的启动标准。

## 停止决定

本轮不运行：

- `bus-text-swe`
- `bus-text-cdfi`
- 其他模态排列组合
- 额外随机种子

下一步不应继续盲目增加模态组合。更合理的方向是先分析同协议全模态在 LA/LB 上的微弱增益是否来自模态交互、类别决策竞争或训练随机性，再决定是否设计任务条件化训练目标。

本报告只记录表示诊断结果，不把层级探针指标当作正式四分类模型性能，也不把它解释为模态因果贡献。
