"""文本语义分支：BioClinicalBERT（冻结）。

直接使用 BioClinicalBERT 提取临床文本特征，不进行微调。
534 样本不足以微调文本分支，冻结 BERT 可避免过拟合，
同时其预训练表征已包含亚型相关语义信息。
"""
import os

# 配置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from transformers import AutoModel


class TextLogicBranch(nn.Module):
    """BioClinicalBERT 文本语义提取分支（完全冻结）。

    BioClinicalBERT 骨干完全冻结（~110M 参数），仅投影层可训练（~0.39M 参数）。
    预训练的 BERT 已能提取亚型相关语义（如 ER/PR/HER2 关键词），
    无需在 534 样本上做 LoRA 微调。
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # 完全冻结 BERT
        for param in self.bert.parameters():
            param.requires_grad = False

        # 文本特征投影到 512 维，用于跨模态融合
        self.projection = nn.Linear(768, 512)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, max_len) — tokenized 临床文本 ID
            attention_mask: (B, max_len) — 注意力掩码

        Returns:
            text_features: (B, 512) — 临床语义特征向量
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        return self.projection(cls_output)  # (B, 512)
