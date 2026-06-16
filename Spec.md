# 面向 AI 辅助编程的底层工程落地规范与分步指南

## 项目：TinySpatial-ContrastNet 多模态乳腺癌亚型诊断框架

为了确保理论架构能够被 AI 编程助手（如 Claude, Copilot）精准转化为生产级代码，本规范定义了组件接口、张量流（Tensor Dataflow）边界及状态机配置，旨在构建高鲁棒性的多模态融合算法，杜绝维度崩塌与梯度回传错误。

---

## 阶段一：项目物理架构与多模态数据管线 (Tensor Factory)

### 1. 物理目录架构规范

底层存储必须严格遵循 `TinySpatial_Project/` 架构，以确保 AI 工具具有准确的路径寻址能力：

```text
TinySpatial_Project/
├── data/
│   ├── images/
│   │   ├── BUS/           # 形态学解剖图 (Gray-scale, 用于 Stage 1 & 2)
│   │   ├── SWE/           # 物理硬度伪彩图 (与 BUS 空间对齐, 用于 Stage 1 & 2)
│   │   └── CDFI/          # 局部血流动力学图 (用于 Stage 2 CNN 独立支路)
│   ├── texts/
│   │   └── {case_id}.json # 临床高阶语义文件 (含 raw_text, 用于 BERT 微调)
│   └── metadata.csv       # 数据引擎"大脑"：统筹 case_id、标签映射与划分策略
├── checkpoints/           # 权重仓库：存放预训练与微调的最佳模型 (.pth)
└── code/                  # 核心系统算法代码目录
```

### 2. 数据读取引擎协议 (MultiModalBreastDataset)

必须继承 `torch.utils.data.Dataset`，通过 `case_id` 实现异构数据的无缝结合。其 `__getitem__` 方法需返回如下结构的字典：

- **`bus_img`**：维度 `1×H×W`，经单通道归一化处理。
- **`swe_img`**：维度 `3×H×W`，剪切波硬度伪彩张量。
- **`cdfi_img`**：维度 `3×H×W`，彩色多普勒血流张量。
- **`clinical_text_tokens`**：经 `transformers.AutoTokenizer` 处理的定长 `input_ids` 与 `attention_mask`。
- **`subtype_label`**：临床分子亚型整数类别 (0-3)。

### 3. 几何一致性增强策略

- **空间对齐约束**：对 `bus_img` 和 `swe_img` 的仿射变换（裁剪、旋转、翻转）必须共用随机数种子，确保特征空间相对坐标绝对静止。
- **独立适配**：`cdfi_img` 允许独立的轻微弹性形变与色彩抖动（Color Jitter），模拟临床按压导致的血流变形。

---

## 阶段二：ACM-MIM 自监督预训练引擎开发

### 1. 联合切片与掩码模块 (JointPatchEmbedding & MaskingEngine)

- **联合输入层**：实例化 `nn.Conv2d(in_channels=4, kernel_size=16, stride=16)`，将 BUS(1ch) + SWE(3ch) 通道拼接后的 4 通道输入下采样为 Patch 序列。内部逻辑：

```python
class JointPatchEmbedding(nn.Module):
    def __init__(self, embed_dim):
        self.projection = nn.Conv2d(in_channels=4, out_channels=embed_dim, kernel_size=16, stride=16)

    def forward(self, bus, swe):
        x = torch.cat([bus, swe], dim=1)  # (B, 4, H, W)
        return self.projection(x).flatten(2).transpose(1, 2)  # (B, N, embed_dim)
```

- **掩码隔离逻辑**：设定掩码比例 $r = 0.75$。核心限制：`mask_matrix` 仅作用于 BUS 通道的重建目标，SWE 对应的 tokens 在编码器中全程保留并参与注意力计算，确保编码器始终拥有完整的力学硬度先验。

### 2. 编码-解码体系

- **Backbone**：实例化 TinyUSFM（5.5M 参数），**此阶段 `requires_grad=True`**，端到端可训练。
- **Decoder**：构建极简 `LightweightDecoder`（全连接或双层 Transformer Block），末端映射回 `16×16×1` 像素空间。
- **损失计算**：仅针对 `mask_matrix == True` 的位置，在预测张量与原始 BUS 张量间执行 `nn.MSELoss()`。

---

## 阶段三：多模态微调架构与解耦注入机制

### 1. 权重加载与深度冻结

将阶段二训练好的 TinyUSFM 权重加载至分型任务，并严格冻结主干：

```python
# 冻结物理表征底层，防止下游梯度破坏
for param in model.backbone.parameters():
    param.requires_grad = False
```

在 800 例样本规模下，冻结主干可有效防止过拟合，保留预训练获得的稳健表征。

### 2. BUS & SWE 分支 (JointPatchEmbedding → Frozen TinyUSFM → [CLS])

原始 BUS 与原始 SWE 经 JointPatchEmbedding 模块（内部 `torch.cat([bus, swe], dim=1)` → `Conv2d(in_channels=4, ...)`，与 Stage 1 结构完全一致）生成 patch tokens，连同 [CLS] token 一起进入 Frozen TinyUSFM。编码器完成全部 12 层前传后，从输出中切取 [CLS] token 作为全局特征 $F_{bus\_swe}$（形态 $B \times 192$）。

两阶段 Patch Embedding 层参数结构一致，确保 Stage 2 可无缝加载 Stage 1 预训练权重。

### 3. CDFI 分支与 IP-Adapter 血流注入 (DecoupledAttentionLayer)

CDFI 图像的特征管线与注入机制如下：

- **CNN 提取器**：使用剥离分类层的 MobileNetV2 提取空间特征图（$B \times C_{cnn} \times H' \times W'$）。
- **Projection Layer**：将 CNN 输出拉平并映射为 **CDFI Tokens**（$B \times N_{cdfi} \times 192$，维度与 TinyUSFM 对齐）。
- **前传入口**：CDFI Tokens 与 [CLS] token 及 BUS+SWE patch tokens **一同进入 TinyUSFM 前传**。
- **IP-Adapter 注入**：在 TinyUSFM **第 6-11 层**，以解耦交叉注意力机制在自注意力旁路**并行添加**注入分支（非重写自注意力）：
  - 保持原有 Key (K) 和 Value (V) 映射锁定不变。
  - 新增可训练线性投影层 $W_k^{cdfi}$ 与 $W_v^{cdfi}$。
  - 编码器内部所有 tokens 作为 Query，CDFI Tokens 作为 Key/Value，使 [CLS] token 逐层吸收血流信息。
  - 因此 $F_{bus\_swe}$ 在提取时已包含 CDFI 语义。
- **动态控制**：暴露 `set_ip_adapter_scale(lambda_scale)` 接口，调节血流特征的话语权。

### 4. 文本语义 LoRA 微调 (TextLogicBranch)

集成 `peft` 框架对 BioClinicalBERT 进行参数高效微调：

- **配置参数**：指定 `target_modules=["query", "value"]`。
- **秩设定**：$r=8$，$\alpha_{lora}=16$。
- **目的**：使模型在有限样本下通过极小增量矩阵捕获临床文本逻辑。

### 5. 跨模态特征融合 (LightweightCrossAttention)

图像与文本特征通过 **LightweightCrossAttention** 模块进行融合（Q=图像, KV=文本），使文本临床语义动态调制图像表征。融合后特征维度为 $\text{image\_dim} + 512$，送入分类头。

### 6. 对比对齐模块 (CAM)

在 **512 维潜空间**内计算图像混合特征与文本向量的 InfoNCE 对比损失 $L_{align}$，消除模型对文本关键词的"捷径学习"。**CAM 仅作为训练正则化项，不参与推理时特征融合**。也可采用去偏 InfoNCE 损失防止"假阴性"惩罚，使相同亚型的病人在潜空间中自然聚类。

**对齐损失 ($L_{align}$):**
$$L_{align} = - \log \frac{\exp(sim(I_i, T_i) / \tau)}{\exp(sim(I_i, T_i) / \tau) + \sum_{j \neq i} \exp(sim(I_i, T_j) / \tau)}$$

**去偏变体：** 引入基于类别先验的校正因子 $q_{ij}$：
$$L_{align}^{debiased} = - \log \frac{\exp(sim(I_i, T_i) / \tau)}{\exp(sim(I_i, T_i) / \tau) + \sum_{j \neq i} q_{ij} \exp(sim(I_i, T_j) / \tau)}$$

### 7. 四分类决策头

最终通过 MLP 输出四种分子亚型的概率分布，输入为 LightweightCrossAttention 融合后的特征（维度 $\text{image\_dim} + 512$）。

---

## 阶段四：训练生命周期管理与临床评估

### 1. 联合优化目标

模型通过加权多任务损失函数进行端到端优化：

$$L_{total} = \alpha L_{cls} + \beta L_{align}$$

其中 **类别加权交叉熵损失 ($L_{cls}$)** 用于解决样本不平衡问题：
$$L_{cls} = - \sum_{i=1}^{4} W_i \cdot y_i \log(\hat{y}_i)$$

- $W_i$：少数派类别加权系数。
- $\hat{y}_i$：分类预测概率。

### 2. 差异化学习率配置组

使用 AdamW 优化器，按组件异构性划分为独立参数群：

| 组件 | 学习率策略 | 目的 |
|------|-----------|------|
| MobileNetV2 (CDFI) | 中等 (Medium) | 快速提取特征分布 |
| IP-Adapter 矩阵 | 较高 (High) | 快速接管血流信息注入 |
| LoRA (BERT) | 较小 (Low) | 防止遗忘预训练文献语义 |
| MLP 分类头 | 正常 (Normal) | 适配最终决策任务 |

配合**余弦退火 (Cosine Annealing)** 引导网络平滑收敛。

### 3. 面向临床价值的性能评估矩阵

集成 `sklearn.metrics` 引擎，除全局 Accuracy 外，必须针对 HER2 过表达型与三阴性 (TNBC) 输出：

- 灵敏度 (Recall / Sensitivity)
- 特异度 (Specificity)
- 阳性预测值 (Precision)
- F1-Score
- 混淆矩阵 (Confusion Matrix)：用于分析各亚型间的误判流向

---

## 系统模块状态规范表

| 模块名称 | 阶段一状态 | 阶段二状态 | 规模 | 处理数据源 | 物理/临床意义 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TinyUSFM** | **可训练** | **冻结 (Frozen)** | ~5.5M | BUS & SWE | 建立解剖与硬度联合感知网络。 |
| **MobileNetV2** | — | 可训练 | ~2.0M | CDFI | 捕捉微血管形态与血流密度特征。 |
| **IP-Adapter** | — | 可训练 | ~1.5M | 血流特征注入 | 实现血流动力学特征的无损融合。 |
| **BioClinicalBERT** | — | **冻结 (Frozen)** | ~110M | 临床文本报告 | 解析医学词元与高阶临床术语。 |
| **LoRA Matrix** | — | 可训练 | ~0.01M | 文本语义微调 | 靶向性提取特定分子亚型描述。 |
| **MLP/CAM** | — | 可训练 | ~0.5M | 多模态融合特征 | 主导高维特征聚类与最终分类决策。 |
