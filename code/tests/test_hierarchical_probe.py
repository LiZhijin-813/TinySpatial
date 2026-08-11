import json
from pathlib import Path

import pytest
import torch

import code.train.run_cv_hierarchical_probe as hierarchical_probe_module

from code.train.run_cv_hierarchical_probe import (
    aggregate_binary_metrics,
    build_hierarchical_tasks,
    run_cv_hierarchical_probe,
    select_hierarchical_labels,
    validate_fold_labels,
)


class _空模型:
    def load_state_dict(self, state_dict, strict=True):
        assert state_dict == {}
        assert strict is True


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _make_run_dir(tmp_path, task_mode="flat5"):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(
        run_dir / "args.json",
        {
            "task_mode": task_mode,
            "malignant_metadata": "metadata.csv",
            "img_size": 224,
            "max_text_len": 32,
            "ablate_modalities": [],
            "num_workers": 0,
        },
    )
    (run_dir / "best_model.pth").write_bytes(b"stub")
    return run_dir


def _make_manifest(tmp_path, folds):
    return _write_json(
        tmp_path / "cv_manifest.json",
        {
            "n_splits": 5,
            "folds": folds,
        },
    )


def _five_single_case_folds(prefix="病例"):
    return [
        {
            "fold": index,
            "case_ids": [f"{prefix}{index}"],
        }
        for index in range(5)
    ]


def _manifest_with(folds=None, n_splits=5):
    if folds is None:
        folds = _five_single_case_folds()
    return {
        "n_splits": n_splits,
        "folds": folds,
    }


def _二十例四分类样本():
    samples = []
    for fold_id in range(5):
        for label in range(4):
            case_id = f"病例{fold_id}-{label}"
            samples.append(
                {
                    "case_id": case_id,
                    "subtype_label": label,
                    "split": "train",
                    "source": "malignant",
                    "class_label": label,
                    "malignancy_label": 1,
                }
            )
    return samples


def _二十例四分类折清单():
    return [
        {
            "fold": fold_id,
            "case_ids": [f"病例{fold_id}-{label}" for label in range(4)],
        }
        for fold_id in range(5)
    ]


def test_validate_manifest_structure_preserves_explicit_fold_ids():
    folds = [
        {"fold": 4, "case_ids": ["case4"]},
        {"fold": 2, "case_ids": ["case2"]},
        {"fold": 0, "case_ids": ["case0"]},
        {"fold": 3, "case_ids": ["case3"]},
        {"fold": 1, "case_ids": ["case1"]},
    ]

    normalized = hierarchical_probe_module._validate_manifest_structure(
        _manifest_with(folds)
    )

    assert [fold["fold"] for fold in normalized] == [4, 2, 0, 3, 1]
    assert sorted(fold["fold"] for fold in normalized) == [0, 1, 2, 3, 4]


def test_validate_manifest_structure_rejects_suspected_group_across_folds():
    folds = _five_single_case_folds(prefix="case")
    folds[0]["case_ids"] = ["1247-L"]
    folds[1]["case_ids"] = ["1247-R"]

    with pytest.raises(ValueError, match="疑似病例组.*1247.*只能属于一个 fold") as exc_info:
        hierarchical_probe_module._validate_manifest_structure(_manifest_with(folds))

    message = str(exc_info.value)
    assert "1247-L" in message
    assert "1247-R" in message
    assert message.isascii() is False


@pytest.mark.parametrize(
    "manifest, error_fragment",
    [
        (_manifest_with(n_splits=4), "n_splits 必须等于 5"),
        ({"n_splits": 5, "folds": "bad"}, "folds 必须是包含五个验证折的列表"),
        (
            {"n_splits": 5, "folds": _five_single_case_folds()[:4]},
            "folds 必须是包含五个验证折的列表",
        ),
        (
            _manifest_with(
                folds=[
                    *_five_single_case_folds()[:2],
                    "bad",
                    *_five_single_case_folds()[3:],
                ]
            ),
            "每个折必须是对象",
        ),
        (
            _manifest_with(
                folds=[
                    {"case_ids": ["case-0"]},
                    *_five_single_case_folds()[1:],
                ]
            ),
            "每个折必须显式包含 fold 字段",
        ),
        (
            _manifest_with(
                folds=[
                    {"fold": True, "case_ids": ["case-0"]},
                    *_five_single_case_folds()[1:],
                ]
            ),
            "fold 字段必须是整数",
        ),
        (
            _manifest_with(
                folds=[
                    {"fold": 0, "case_ids": "case-0"},
                    *_five_single_case_folds()[1:],
                ]
            ),
            "case_ids 必须是非空列表",
        ),
        (
            _manifest_with(
                folds=[
                    {"fold": 0, "case_ids": []},
                    *_five_single_case_folds()[1:],
                ]
            ),
            "case_ids 必须是非空列表",
        ),
        (
            _manifest_with(
                folds=[
                    {"fold": 0, "case_ids": ["case-0", " "]},
                    *_five_single_case_folds()[1:],
                ]
            ),
            "case_ids 包含无效病例编号",
        ),
        (
            _manifest_with(
                folds=[
                    {"fold": 0, "case_ids": ["case-0", "case-0"]},
                    *_five_single_case_folds()[1:],
                ]
            ),
            "case_ids 包含重复病例编号",
        ),
        (
            _manifest_with(
                folds=[
                    {"fold": 0, "case_ids": ["case-0"]},
                    {"fold": 1, "case_ids": ["case-1"]},
                    {"fold": 1, "case_ids": ["case-2"]},
                    {"fold": 3, "case_ids": ["case-3"]},
                    {"fold": 4, "case_ids": ["case-4"]},
                ]
            ),
            "fold 字段必须唯一且集合正好为 0..4",
        ),
        (
            _manifest_with(
                folds=[
                    *_five_single_case_folds()[:4],
                    {"fold": 5, "case_ids": ["case-4"]},
                ]
            ),
            "fold 字段必须唯一且集合正好为 0..4",
        ),
    ],
)
def test_validate_manifest_structure_reports_chinese_errors(
    manifest, error_fragment
):
    with pytest.raises(ValueError, match=error_fragment) as exc_info:
        hierarchical_probe_module._validate_manifest_structure(manifest)

    assert str(exc_info.value).isascii() is False


def test_hierarchical_probe_tests_read_output_path_as_utf8():
    source_text = Path(__file__).read_text(encoding="utf-8")
    bad_output_text = bytes.fromhex("e69d88e692b3e59aad").decode("utf-8")

    assert 'tmp_path / "输出"' in source_text
    assert bad_output_text not in source_text


def test_normalize_manifest_folds_requires_explicit_fold_field():
    folds = [
        {"case_ids": ["case-0"]},
        *_five_single_case_folds()[1:],
    ]
    case_label_map = {f"case{index}": index % 4 for index in range(5)}

    with pytest.raises(ValueError, match="每个折必须显式包含 fold 字段") as exc_info:
        hierarchical_probe_module._normalize_manifest_folds(
            _manifest_with(folds),
            case_label_map,
        )

    assert str(exc_info.value).isascii() is False


@pytest.mark.parametrize(
    "folds, case_label_map, error_fragment",
    [
        (
            [
                *_five_single_case_folds()[:4],
                {"fold": 4, "case_ids": ["unknown-case"]},
            ],
            {f"病例{index}": index % 4 for index in range(5)},
            "未知病例编号",
        ),
        (
            _five_single_case_folds(prefix="case"),
            {f"case{index}": index % 4 for index in range(6)},
            "未覆盖全部恶性病例",
        ),
    ],
)
def test_normalize_manifest_folds_reports_chinese_cross_case_errors(
    folds, case_label_map, error_fragment
):
    with pytest.raises(ValueError, match=error_fragment) as exc_info:
        hierarchical_probe_module._normalize_manifest_folds(
            _manifest_with(folds),
            case_label_map,
        )

    assert str(exc_info.value).isascii() is False


def test_normalize_manifest_folds_rejects_suspected_group_across_folds():
    folds = _five_single_case_folds(prefix="case")
    folds[0]["case_ids"] = ["1247-L"]
    folds[1]["case_ids"] = ["1247-R"]
    case_label_map = {
        "1247-L": 0,
        "1247-R": 1,
        "case2": 2,
        "case3": 3,
        "case4": 0,
    }

    with pytest.raises(ValueError, match="疑似病例组.*1247.*只能属于一个 fold") as exc_info:
        hierarchical_probe_module._normalize_manifest_folds(
            _manifest_with(folds),
            case_label_map,
        )

    message = str(exc_info.value)
    assert "1247-L" in message
    assert "1247-R" in message
    assert message.isascii() is False


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


def test_aggregate_binary_metrics_summarizes_complete_metrics_and_per_class():
    aggregated = aggregate_binary_metrics(
        [
            {
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "macro_precision": 0.25,
                "macro_recall": 0.5,
                "macro_f1": 0.3333333333,
                "weighted_f1": 0.3333333333,
                "per_class": {
                    "Luminal": {
                        "recall": 1.0,
                        "precision": 0.5,
                        "specificity": 0.0,
                        "f1": 2 / 3,
                    },
                    "Non-Luminal": {
                        "recall": 0.0,
                        "precision": 0.0,
                        "specificity": 1.0,
                        "f1": 0.0,
                    },
                },
            },
            {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_precision": 1.0,
                "macro_recall": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
                "per_class": {
                    "Luminal": {
                        "recall": 1.0,
                        "precision": 1.0,
                        "specificity": 1.0,
                        "f1": 1.0,
                    },
                    "Non-Luminal": {
                        "recall": 1.0,
                        "precision": 1.0,
                        "specificity": 1.0,
                        "f1": 1.0,
                    },
                },
            },
        ]
    )

    assert set(aggregated) == {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "per_class",
    }
    assert aggregated["accuracy"] == {
        "mean": 0.75,
        "std": 0.25,
        "values": [0.5, 1.0],
    }
    assert aggregated["macro_precision"] == {
        "mean": 0.625,
        "std": 0.375,
        "values": [0.25, 1.0],
    }
    assert aggregated["per_class"]["Luminal"]["precision"] == {
        "mean": 0.75,
        "std": 0.25,
        "values": [0.5, 1.0],
    }
    assert aggregated["per_class"]["Non-Luminal"]["recall"] == {
        "mean": 0.5,
        "std": 0.5,
        "values": [0.0, 1.0],
    }
    assert aggregated["per_class"]["Non-Luminal"]["specificity"] == {
        "mean": 1.0,
        "std": 0.0,
        "values": [1.0, 1.0],
    }


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


def test_validate_fold_labels_rejects_empty_query_labels():
    with pytest.raises(ValueError, match="缺少二分类类别"):
        validate_fold_labels(
            torch.tensor([0, 1]),
            torch.empty(0, dtype=torch.long),
            "示例任务",
        )


def test_run_cv_hierarchical_probe_rejects_non_five_fold_manifest_without_loading_weights(
    tmp_path, monkeypatch
):
    run_dir = _make_run_dir(tmp_path)
    manifest_path = _make_manifest(tmp_path, _five_single_case_folds()[:4])
    output_dir = tmp_path / "输出"
    output_dir.mkdir()

    monkeypatch.setattr(
        hierarchical_probe_module.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("边界测试不应加载检查点权重"),
    )

    with pytest.raises(ValueError, match="五个验证折"):
        run_cv_hierarchical_probe(
            run_dir,
            manifest_path,
            output_dir,
            device_name="cpu",
            batch_size=2,
        )


def test_run_cv_hierarchical_probe_rejects_flat4_checkpoint_without_loading_weights(
    tmp_path, monkeypatch
):
    run_dir = _make_run_dir(tmp_path, task_mode="flat4")
    manifest_path = _make_manifest(tmp_path, _five_single_case_folds())
    output_dir = tmp_path / "输出"
    output_dir.mkdir()

    monkeypatch.setattr(
        hierarchical_probe_module.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("边界测试不应加载检查点权重"),
    )

    with pytest.raises(ValueError, match="flat5"):
        run_cv_hierarchical_probe(
            run_dir,
            manifest_path,
            output_dir,
            device_name="cpu",
            batch_size=2,
        )


@pytest.mark.parametrize(
    "folds, error_fragment",
    [
        (
            [
                {"fold": 0, "case_ids": ["case-0"]},
                {"fold": 1, "case_ids": ["case-1"]},
                {"fold": 1, "case_ids": ["case-2"]},
                {"fold": 3, "case_ids": ["case-3"]},
                {"fold": 4, "case_ids": ["case-4"]},
            ],
            "0..4",
        ),
        (
            [
                {"fold": 0, "case_ids": ["case-0"]},
                {"fold": 1, "case_ids": ["case-1"]},
                {"fold": 2, "case_ids": ["case-2"]},
                {"fold": 3, "case_ids": ["case-3"]},
                {"fold": 5, "case_ids": ["case-4"]},
            ],
            "0..4",
        ),
    ],
)
def test_run_cv_hierarchical_probe_rejects_invalid_fold_ids_before_metadata_or_weights(
    tmp_path, monkeypatch, folds, error_fragment
):
    run_dir = _make_run_dir(tmp_path)
    manifest_path = _make_manifest(tmp_path, folds)
    output_dir = tmp_path / "输出"
    output_dir.mkdir()

    monkeypatch.setattr(
        hierarchical_probe_module,
        "load_metadata",
        lambda *args, **kwargs: pytest.fail("无效折编号不应加载元数据"),
    )
    monkeypatch.setattr(
        hierarchical_probe_module.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("无效折编号不应加载权重"),
    )

    with pytest.raises(ValueError, match=error_fragment):
        run_cv_hierarchical_probe(
            run_dir,
            manifest_path,
            output_dir,
            device_name="cpu",
            batch_size=2,
        )


def test_run_cv_hierarchical_probe_rejects_suspected_group_before_metadata_or_weights(
    tmp_path, monkeypatch
):
    run_dir = _make_run_dir(tmp_path)
    folds = _five_single_case_folds(prefix="case")
    folds[0]["case_ids"] = ["1247-L"]
    folds[1]["case_ids"] = ["1247-R"]
    manifest_path = _make_manifest(tmp_path, folds)
    output_dir = tmp_path / "输出"
    output_dir.mkdir()

    monkeypatch.setattr(
        hierarchical_probe_module,
        "load_metadata",
        lambda *args, **kwargs: pytest.fail("疑似病例组跨折不应加载元数据"),
    )
    monkeypatch.setattr(
        hierarchical_probe_module.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("疑似病例组跨折不应加载权重"),
    )

    with pytest.raises(ValueError, match="疑似病例组.*1247.*只能属于一个 fold"):
        run_cv_hierarchical_probe(
            run_dir,
            manifest_path,
            output_dir,
            device_name="cpu",
            batch_size=2,
        )


def test_run_cv_hierarchical_probe_rejects_empty_hierarchical_selection_without_loading_weights(
    tmp_path, monkeypatch
):
    run_dir = _make_run_dir(tmp_path)
    manifest_path = _make_manifest(tmp_path, _five_single_case_folds())
    output_dir = tmp_path / "输出"
    output_dir.mkdir()
    tasks = build_hierarchical_tasks()

    monkeypatch.setattr(
        hierarchical_probe_module,
        "build_hierarchical_tasks",
        lambda: {"her2_vs_tnbc": tasks["her2_vs_tnbc"]},
    )
    monkeypatch.setattr(
        hierarchical_probe_module,
        "load_metadata",
        lambda *args, **kwargs: [
            {
                "case_id": f"病例{index}",
                "subtype_label": index % 2,
                "split": "train",
                "source": "malignant",
                "class_label": index % 2,
                "malignancy_label": 1,
            }
            for index in range(5)
        ],
    )
    monkeypatch.setattr(
        hierarchical_probe_module.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("边界测试不应加载检查点权重"),
    )

    with pytest.raises(ValueError, match="层级筛选结果为空"):
        run_cv_hierarchical_probe(
            run_dir,
            manifest_path,
            output_dir,
            device_name="cpu",
            batch_size=2,
        )


def test_run_cv_hierarchical_probe_cpu_mock_success_writes_task_protocol(
    tmp_path, monkeypatch
):
    run_dir = _make_run_dir(tmp_path)
    manifest_path = _make_manifest(tmp_path, _二十例四分类折清单())
    output_dir = tmp_path / "输出"
    samples = _二十例四分类样本()
    labels = torch.tensor([sample["subtype_label"] for sample in samples], dtype=torch.long)
    features = torch.arange(len(samples) * 3, dtype=torch.float32).reshape(len(samples), 3)
    case_ids = [sample["case_id"] for sample in samples]

    monkeypatch.setattr(
        hierarchical_probe_module,
        "load_metadata",
        lambda *args, **kwargs: samples,
    )
    monkeypatch.setattr(
        hierarchical_probe_module,
        "MultiModalBreastDataset",
        lambda *args, **kwargs: {"samples": kwargs["samples"]},
    )
    monkeypatch.setattr(
        hierarchical_probe_module,
        "build_eval_loader",
        lambda dataset, loader_args: ("loader", dataset, loader_args.batch_size),
    )
    monkeypatch.setattr(
        hierarchical_probe_module,
        "build_model",
        lambda model_args, device: _空模型(),
    )
    monkeypatch.setattr(
        hierarchical_probe_module.torch,
        "load",
        lambda *args, **kwargs: {"model_state_dict": {}},
    )
    monkeypatch.setattr(
        hierarchical_probe_module,
        "_collect_features",
        lambda model, loader, device: (features, labels, case_ids),
    )
    monkeypatch.setattr(
        hierarchical_probe_module,
        "fit_linear_probe",
        lambda train_features, train_labels, query_features, **kwargs: torch.arange(
            query_features.shape[0],
            dtype=torch.long,
        )
        % 2,
    )

    results = run_cv_hierarchical_probe(
        run_dir,
        manifest_path,
        output_dir,
        device_name="cpu",
        batch_size=4,
    )

    output_file = output_dir / "hierarchical_probe_metrics.json"
    assert output_file.is_file()
    saved_results = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved_results == results
    assert saved_results["sample_count"] == 20
    assert saved_results["feature_dimension"] == 3
    assert "仅使用训练折拟合二分类线性探针" in saved_results["protocol"]

    luminal_task = saved_results["tasks"]["luminal_vs_non_luminal"]
    assert luminal_task["effective_count"] == 20
    assert luminal_task["label_distribution"] == [10, 10]
    assert luminal_task["fold_query_effective_counts"] == [
        {"fold": 0, "query_effective_count": 4},
        {"fold": 1, "query_effective_count": 4},
        {"fold": 2, "query_effective_count": 4},
        {"fold": 3, "query_effective_count": 4},
        {"fold": 4, "query_effective_count": 4},
    ]

    luminal_a_task = saved_results["tasks"]["luminal_a_vs_luminal_b"]
    assert luminal_a_task["effective_count"] == 10
    assert luminal_a_task["label_distribution"] == [5, 5]
    assert luminal_a_task["fold_query_effective_counts"] == [
        {"fold": fold_id, "query_effective_count": 2}
        for fold_id in range(5)
    ]

    for task_result in saved_results["tasks"].values():
        assert [record["fold"] for record in task_result["methods"]["plain"]] == [
            0,
            1,
            2,
            3,
            4,
        ]
        assert set(task_result["summary"]) == {"plain", "inverse_frequency"}
        assert "macro_f1" in task_result["summary"]["plain"]
        first_record = task_result["methods"]["plain"][0]
        assert first_record["confusion_matrix"]
        assert first_record["train_effective_count"] > 0
        assert first_record["query_effective_count"] > 0
