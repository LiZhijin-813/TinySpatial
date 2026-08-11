import pytest
import torch

from code.train.run_cv_hierarchical_probe import (
    build_hierarchical_tasks,
    select_hierarchical_labels,
)


def test_层级任务内部筛选会排除无关类别():
    tasks = build_hierarchical_tasks()
    mask, binary = select_hierarchical_labels(
        torch.tensor([0, 1, 2, 3, 0]),
        tasks["luminal_a_vs_luminal_b"],
    )
    assert mask.tolist() == [True, True, False, False, True]
    assert binary.tolist() == [0, 1, 0]


def test_层级任务会拒绝四类空间之外的标签():
    tasks = build_hierarchical_tasks()
    with pytest.raises(ValueError, match="标签"):
        select_hierarchical_labels(
            torch.tensor([0, 4]),
            tasks["luminal_vs_non_luminal"],
        )
