"""随机种子与可复现性工具模块。

统一设置 Python、NumPy、PyTorch 的随机种子，
确保跨框架实验结果可复现。
"""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """设置所有框架的随机种子以确保实验可复现。

    Args:
        seed: 随机种子值，默认为 42
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确定性模式：牺牲部分性能换取可复现性
    torch.backends.cudnn.deterministic = True
    # 关闭 cuDNN 自动调优，避免不同运行间选择不同算法
    torch.backends.cudnn.benchmark = False
