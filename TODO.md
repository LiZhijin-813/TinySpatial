# TinySpatial-ContrastNet 开发任务清单

基于 `Spec.md` 与 `Readme.md` 的工程落地分解，按依赖顺序排列。

---

## Phase 0：项目基础设施

- [ ] **P0-1** 搭建物理目录架构：`data/images/{BUS,SWE,CDFI}`, `data/texts/`, `checkpoints/`, `code/`
- [ ] **P0-2** 初始化 `code/` 下的 Python 包结构（`__init__.py`、`requirements.txt` / `pyproject.toml`）
- [ ] **P0-3** 配置随机种子与可复现性工具（`torch.manual_seed`, `cudnn.deterministic`）
已完成
---

## Phase 1：数据管线

- [ ] **P1-1** 实现 `MultiModalBreastDataset`（继承 `torch.utils.data.Dataset`）
  - `__getitem__` 返回：`bus_img(1×H×W)`, `swe_img(3×H×W)`, `cdfi_img(3×H×W)`, `clinical_text_tokens(input_ids, attention_mask)`, `subtype_label(0-3)`
  - 通过 `case_id` 从 `metadata.csv` 索引并关联各模态文件
- [ ] **P1-2** 实现空间对齐增强：`bus_img` 与 `swe_img` 共用随机种子执行仿射变换（裁剪、旋转、翻转）
- [ ] **P1-3** 实现独立适配增强：`cdfi_img` 的弹性形变与 Color Jitter
- [ ] **P1-4** 集成 `transformers.AutoTokenizer` 对临床文本进行定长 tokenization
- [ ] **P1-5** 编写数据管线单元测试：验证输出维度、增强一致性（BUS-SWE 对齐验证）、tokenization 正确性

**依赖**：P0-1 → P1-1 → P1-2, P1-3, P1-4 → P1-5

---

## Phase 2：ACM-MIM 自监督预训练

- [ ] **P2-1** 实现 `JointPatchEmbedding`
  - `Conv2d(in_channels=4, kernel_size=16, stride=16)`，`forward` 内部 `torch.cat([bus, swe], dim=1)`
  - 输出：(B, N, embed_dim)
- [ ] **P2-2** 实现 `MaskingEngine`
  - 生成 75% 随机掩码矩阵，仅作用于 BUS 通道重建目标
  - SWE tokens 全程保留，不参与掩码
- [ ] **P2-3** 集成 TinyUSFM 编码器
  - 加载预训练权重：`TinySpatial_Project/TinyUSFM.pth`
  - Stage 1 可训练，`requires_grad=True`
- [ ] **P2-4** 实现 `LightweightDecoder`（全连接或双层 Transformer Block），末端映射回 `16×16×1` 像素空间
- [ ] **P2-5** 实现重建损失：仅对 `mask_matrix == True` 位置计算 MSE Loss
- [ ] **P2-6** 组装 Stage 1 完整前传流程：输入 → JointPatchEmbedding → MaskingEngine → TinyUSFM → Decoder → Loss
- [ ] **P2-7** 编写 Stage 1 训练脚本：含 checkpoint 保存逻辑
- [ ] **P2-8** 编写 Stage 1 维度验证测试：输入 (B,4,H,W) → 编码器输出 → Decoder 输出 → 损失标量

**依赖**：P1-1, P1-2 → P2-1, P2-2 → P2-3 → P2-4 → P2-5 → P2-6 → P2-7, P2-8

---

## Phase 3：多模态微调架构

- [ ] **P3-1** 实现 Stage 2 权重加载与冻结逻辑
  - 加载预训练权重：`TinySpatial_Project/TinyUSFM.pth`
  - `model.backbone.parameters()` 全部 `requires_grad=False`
- [ ] **P3-2** 实现 CDFI 特征提取管线
  - MobileNetV2（剥离分类层）→ 空间特征图 ($B \times C_{cnn} \times H' \times W'$)
  - Projection Layer → CDFI Tokens ($B \times N_{cdfi} \times 192$)
- [ ] **P3-3** 实现 `DecoupledAttentionLayer`（IP-Adapter 核心模块）
  - 保持原有 K/V 映射锁定，新增 $W_k^{cdfi}$、$W_v^{cdfi}$ 线性投影层
  - 主干 Q 与固有 K、新增 $K_{cdfi}$ 分别计算内积，结果并行相加
  - `set_ip_adapter_scale(lambda_scale)` 接口
- [ ] **P3-4** 将 IP-Adapter 注入 TinyUSFM 第 6-11 层
  - CDFI Tokens 与 [CLS] + patch tokens 一同进入前传
  - 第 6-11 层激活解耦交叉注意力，[CLS] 逐层吸收血流信息
- [ ] **P3-5** 实现 `TextLogicBranch`：BioClinicalBERT + LoRA
  - 使用 `peft` 框架，`target_modules=["query", "value"]`，$r=8$，$\alpha_{lora}=16$
- [ ] **P3-6** 实现 `LightweightCrossAttention`
  - Q=图像特征（$F_{bus\_swe}$, $B \times 192$），KV=文本特征（$F_{text}$）
  - 融合后维度：image_dim + 512
- [ ] **P3-7** 实现四分类 MLP 决策头
  - 输入维度：image_dim + 512，输出 4 类概率
- [ ] **P3-8** 组装 Stage 2 完整模型前传：三模态输入 → 各支路提取 → IP-Adapter 注入 → LightweightCrossAttention → MLP → 输出
- [ ] **P3-9** 编写 Stage 2 维度验证测试：各支路输出形态、IP-Adapter 注入前后维度一致性、最终 MLP 输出 4 类

**依赖**：P2-7 → P3-1；P3-2, P3-3 → P3-4 → P3-8；P3-5, P3-6, P3-7 → P3-8 → P3-9

---

## Phase 4：对比对齐模块 (CAM)

- [ ] **P4-1** 实现 512 维投影头（图像侧 + 文本侧各一个线性层，映射至 512 维潜空间）
- [ ] **P4-2** 实现 InfoNCE 对比损失
  - 温度参数 $\tau$ 可配置
  - 图像-文本配对，batch 内负样本
- [ ] **P4-3** 实现去偏 InfoNCE 变体（可选）
  - 基于类别先验的校正因子 $q_{ij}$
- [ ] **P4-4** 确认 CAM 仅参与训练损失计算，不参与推理路径
- [ ] **P4-5** 编写 CAM 测试：投影维度验证、损失值范围合理性

**依赖**：P3-8 → P4-1 → P4-2 → P4-3 → P4-5

---

## Phase 5：训练生命周期

- [ ] **P5-1** 实现联合优化目标：$L_{total} = \alpha L_{cls} + \beta L_{align}$
- [ ] **P5-2** 实现类别加权交叉熵损失 $L_{cls}$（根据数据分布计算 $W_i$）
- [ ] **P5-3** 实现 AdamW 差异化学习率配置
  - MobileNetV2: Medium | IP-Adapter: High | LoRA: Low | MLP: Normal
- [ ] **P5-4** 实现 Cosine Annealing 学习率调度
- [ ] **P5-5** 实现 Stage 2 完整训练脚本
  - 含训练/验证循环、early stopping、最佳模型保存
- [ ] **P5-6** 编写训练冒烟测试：1 个 epoch 确保无 NaN/维度错误，梯度正常回传

**依赖**：P3-8, P4-2 → P5-1 → P5-2, P5-3, P5-4 → P5-5 → P5-6

---

## Phase 6：临床评估与工具链

- [ ] **P6-1** 实现评估管线：Accuracy, Per-class Recall, Specificity, Precision, F1-Score
- [ ] **P6-2** 实现混淆矩阵可视化，分析各亚型间误判流向
- [ ] **P6-3** 特别关注 HER2+ 与 TNBC 的分型指标
- [ ] **P6-4** 实现推理脚本：单例预测 + 批量评估
- [ ] **P6-5** 编写端到端集成测试：数据加载 → 模型推理 → 评估指标输出

**依赖**：P5-5 → P6-1 → P6-2, P6-3 → P6-4 → P6-5

---

## 并行化建议

```
P0 (基础设施) ──→ P1 (数据管线) ──→ P2 (Stage 1 预训练)
                                           │
                                           ↓
                                      P3 (Stage 2 架构)
                                       │        │
                                       ↓        ↓
                                   P4 (CAM)   P5 (训练)
                                       │        │
                                       ↓        ↓
                                      P6 (评估)
```

- **P3-2 (CDFI 管线)** 与 **P3-5 (文本 LoRA)** 可并行开发
- **P4 (CAM)** 与 **P5-2, P5-3, P5-4 (损失/优化器)** 可并行开发
