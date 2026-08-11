import pytest
import torch

from code.train.run_cv_hierarchical_probe import (
    aggregate_binary_metrics,
    build_hierarchical_tasks,
    select_hierarchical_labels,
    validate_fold_labels,
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


def test_select_hierarchical_labels_filters_unrelated_classes():
    tasks = build_hierarchical_tasks()
    mask, binary = select_hierarchical_labels(
        torch.tensor([0, 1, 2, 3, 0]),
        tasks["luminal_a_vs_luminal_b"],
    )
    assert mask.tolist() == [True, True, False, False, True]
    assert binary.tolist() == [0, 1, 0]


def test_select_hierarchical_labels_rejects_out_of_range_labels():
    tasks = build_hierarchical_tasks()
    with pytest.raises(ValueError, match="标签"):
        select_hierarchical_labels(
            torch.tensor([0, 4]),
            tasks["luminal_vs_non_luminal"],
        )


@pytest.mark.parametrize(
    "labels, error_fragment",
    [
        (torch.tensor([[0, 1]]), "一维"),
        (torch.tensor([0.0, 1.0]), "整数"),
        (torch.tensor([-1, 1]), "四分类空间"),
    ],
)
def test_select_hierarchical_labels_rejects_invalid_input(
    labels, error_fragment
):
    tasks = build_hierarchical_tasks()

    with pytest.raises(ValueError, match=error_fragment):
        select_hierarchical_labels(labels, tasks["luminal_vs_non_luminal"])


def test_aggregate_binary_metrics_returns_mean_std_and_values():
    aggregated = aggregate_binary_metrics(
        [
            {
                "macro_f1": 0.25,
                "balanced_accuracy": 0.5,
            },
            {
                "macro_f1": 0.75,
                "balanced_accuracy": 1.0,
            },
        ]
    )

    assert aggregated == {
        "macro_f1": {
            "mean": 0.5,
            "std": 0.25,
            "values": [0.25, 0.75],
        },
        "balanced_accuracy": {
            "mean": 0.75,
            "std": 0.25,
            "values": [0.5, 1.0],
        },
    }
    assert all(
        isinstance(value, float)
        for metric in aggregated.values()
        for value in [metric["mean"], metric["std"], *metric["values"]]
    )


def test_aggregate_binary_metrics_rejects_empty_metrics():
    with pytest.raises(ValueError, match="不能为空"):
        aggregate_binary_metrics([])


def test_validate_fold_labels_accepts_binary_integer_labels():
    validate_fold_labels(
        torch.tensor([0, 1, 0, 1], dtype=torch.int64),
        torch.tensor([1, 0], dtype=torch.int32),
        "示例任务",
    )


@pytest.mark.parametrize(
    "train_labels, query_labels, error_fragment",
    [
        (
            torch.tensor([[0, 1]], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
            "一维整数张量",
        ),
        (
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.int64),
            "一维整数张量",
        ),
        (
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([[0, 1]], dtype=torch.int64),
            "一维整数张量",
        ),
        (
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            "一维整数张量",
        ),
    ],
)
def test_validate_fold_labels_rejects_non_vector_or_non_integer_tensors(
    train_labels, query_labels, error_fragment
):
    with pytest.raises(ValueError, match=error_fragment):
        validate_fold_labels(train_labels, query_labels, "示例任务")


def test_validate_fold_labels_rejects_train_labels_without_both_classes():
    with pytest.raises(ValueError, match="缺少二分类类别"):
        validate_fold_labels(
            torch.tensor([0, 0, 0], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
            "示例任务",
        )


def test_validate_fold_labels_rejects_query_labels_without_both_classes():
    with pytest.raises(ValueError, match="缺少二分类类别"):
        validate_fold_labels(
            torch.tensor([0, 1, 0], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
            "示例任务",
        )

def test_validate_fold_labels_rejects_empty_training_labels():
    with pytest.raises(ValueError, match="缺少二分类类别"):
        validate_fold_labels(
            torch.empty(0, dtype=torch.long),
            torch.tensor([0, 1]),
            "示例任务",
        )
