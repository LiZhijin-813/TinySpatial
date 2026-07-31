"""Stage 2 平坦分类头与层级双任务头。"""

import torch.nn as nn


def _mlp(input_dim, output_dim, hidden_dim):
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, output_dim),
    )


class Stage2TaskHeads(nn.Module):
    """根据任务模式输出平坦 logits 或良恶性与亚型双头 logits。"""

    def __init__(self, input_dim: int, task_mode: str, hidden_dim: int = 128):
        super().__init__()
        if task_mode not in {"flat4", "flat5", "dual_head"}:
            raise ValueError(f"任务分类头不支持模式: {task_mode}")

        self.task_mode = task_mode
        if task_mode == "dual_head":
            self.malignancy_head = _mlp(input_dim, 2, hidden_dim)
            self.subtype_head = _mlp(input_dim, 4, hidden_dim)
        else:
            self.class_head = _mlp(
                input_dim,
                4 if task_mode == "flat4" else 5,
                hidden_dim,
            )

    def forward(self, features):
        if self.task_mode == "dual_head":
            return {
                "malignancy_logits": self.malignancy_head(features),
                "subtype_logits": self.subtype_head(features),
            }
        return {"class_logits": self.class_head(features)}
