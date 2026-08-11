import pytest
import torch

from code.train.run_cv_hierarchical_probe import (
    build_hierarchical_tasks,
    select_hierarchical_labels,
)


def test_build_hierarchical_tasks_returns_three_expected_tasks():
    tasks = build_hierarchical_tasks()

    assert set(tasks) == {
        "luminal_vs_non_luminal",
        "luminal_a_vs_luminal_b",
        "her2_vs_tnbc",
    }
    assert tasks["luminal_vs_non_luminal"] == {
        "positive_labels": (0, 1),
        "negative_labels": (2, 3),
        "display_names": ("Luminal", "非Luminal"),
    }
    assert tasks["luminal_a_vs_luminal_b"] == {
        "positive_labels": (0,),
        "negative_labels": (1,),
        "display_names": ("Luminal A", "Luminal B"),
    }
    assert tasks["her2_vs_tnbc"] == {
        "positive_labels": (2,),
        "negative_labels": (3,),
        "display_names": ("HER2+", "TNBC"),
    }


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


@pytest.mark.parametrize(
    "labels, 错误片段",
    [
        (torch.tensor([[0, 1]]), "一维"),
        (torch.tensor([0.0, 1.0]), "整数"),
        (torch.tensor([-1, 1]), "四分类空间"),
    ],
)
def test_层级任务会拒绝非法标签输入(labels, 错误片段):
    tasks = build_hierarchical_tasks()

    with pytest.raises(ValueError, match=错误片段):
        select_hierarchical_labels(labels, tasks["luminal_vs_non_luminal"])
