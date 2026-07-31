"""定义 Stage 2 JSON 运行产物与模型选择指标契约。"""

import json
import re
from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import RandomSampler, WeightedRandomSampler

from code.train import train_stage2 as train_stage2_module
from code.train.run_artifacts import (
    initialize_run_artifacts,
    save_json,
    save_training_state,
)
from code.train.stage2_engine import monitor_value
from code.train.train_stage2 import (
    assert_overfit_gate,
    build_criteria_for_mode,
    build_optimizer,
    build_parser,
    build_train_loader,
    evaluate_checkpoint,
    restore_manifest_splits,
    validate_reliable_configuration,
)


def _assert_chinese_value_error(error):
    """确认异常类型之外还提供可读中文错误信息。"""
    assert re.search(r"[\u4e00-\u9fff]", str(error.value))


def test_run_artifacts_write_required_json_files(tmp_path):
    """初始化与训练状态保存必须生成四个约定的 JSON 文件。"""
    args = Namespace(task_mode="dual_head", seed=42)
    manifest = {"task_mode": "dual_head", "splits": {"train": ["001"]}}

    initialize_run_artifacts(tmp_path, args, manifest)
    save_training_state(
        tmp_path,
        history=[{"epoch": 1, "train": {"total_loss": 1.0}}],
        best_metrics={"malignant_macro_f1": 0.4},
    )

    assert json.loads(
        (tmp_path / "args.json").read_text(encoding="utf-8")
    ) == vars(args)
    assert json.loads(
        (tmp_path / "split_manifest.json").read_text(encoding="utf-8")
    ) == manifest
    assert json.loads(
        (tmp_path / "history.json").read_text(encoding="utf-8")
    ) == [{"epoch": 1, "train": {"total_loss": 1.0}}]
    assert json.loads(
        (tmp_path / "metrics_best.json").read_text(encoding="utf-8")
    ) == {"malignant_macro_f1": 0.4}


def test_json_artifacts_serialize_namespace_numpy_and_torch_scalars(tmp_path):
    """Namespace、NumPy 标量与 Torch 标量必须可写入标准 JSON。"""
    args = Namespace(
        seed=np.int64(7),
        threshold=torch.tensor(0.25),
    )
    manifest = {
        "fold": np.int32(2),
        "score": np.float32(0.5),
        "count": torch.tensor(3),
    }

    initialize_run_artifacts(tmp_path, args, manifest)

    saved_args = json.loads((tmp_path / "args.json").read_text(encoding="utf-8"))
    saved_manifest = json.loads(
        (tmp_path / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_args == {"seed": 7, "threshold": pytest.approx(0.25)}
    assert saved_manifest == {
        "fold": 2,
        "score": pytest.approx(0.5),
        "count": 3,
    }


def test_save_json_preserves_utf8_chinese(tmp_path):
    """JSON 文件必须保留中文原文而不是写成 Unicode 转义。"""
    path = tmp_path / "中文产物.json"

    save_json(path, {"病例": "恶性亚型", "说明": "可审计训练"})

    raw_text = path.read_text(encoding="utf-8")
    assert "病例" in raw_text
    assert "恶性亚型" in raw_text
    assert "\\u75c5\\u4f8b" not in raw_text
    assert json.loads(raw_text)["说明"] == "可审计训练"


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float32(np.nan),
        np.array([1.0, np.inf]),
        torch.tensor(float("nan")),
        torch.tensor([1.0, float("inf")]),
    ],
)
def test_save_json_rejects_non_finite_numbers(tmp_path, value):
    """严格 JSON 必须拒绝 Python、NumPy 与 Torch 中的 NaN 和 Inf。"""
    path = tmp_path / "invalid.json"

    with pytest.raises(ValueError) as error:
        save_json(path, {"value": value})

    _assert_chinese_value_error(error)
    assert not path.exists()


def test_monitor_metric_contract_uses_named_sections():
    """显式诊断模式下兼容监控项必须从约定指标分区读取数值。"""
    metrics = {
        "malignant": {
            "macro_f1": 0.41,
            "balanced_accuracy": 0.43,
            "accuracy": 0.45,
        },
        "overall": {
            "macro_f1": 0.81,
            "balanced_accuracy": 0.82,
            "accuracy": 0.72,
        },
        "binary": {"macro_f1": 0.91, "accuracy": 0.92},
    }

    assert monitor_value(metrics, "malignant_macro_f1") == pytest.approx(0.41)
    assert monitor_value(
        metrics,
        "macro_f1",
        allow_diagnostic=True,
    ) == pytest.approx(0.41)
    assert monitor_value(
        metrics,
        "balanced_acc",
        allow_diagnostic=True,
    ) == pytest.approx(0.43)
    assert monitor_value(
        metrics,
        "acc",
        allow_diagnostic=True,
    ) == pytest.approx(0.72)


@pytest.mark.parametrize("monitor_metric", ["macro_f1", "balanced_acc", "acc"])
def test_diagnostic_monitor_metrics_require_explicit_opt_in(monitor_metric):
    """默认模型选择必须拒绝所有诊断监控项。"""
    metrics = {
        "malignant": {
            "macro_f1": 0.41,
            "balanced_accuracy": 0.43,
            "accuracy": 0.45,
        },
        "overall": {"accuracy": 0.72},
    }

    with pytest.raises(ValueError) as error:
        monitor_value(metrics, monitor_metric)

    _assert_chinese_value_error(error)


def test_malignant_macro_f1_reads_only_malignant_macro_f1():
    """主模型选择值不得读取 flat5 总体准确率或其他宏平均指标。"""
    metrics = {
        "malignant": {"macro_f1": np.float32(0.37)},
        "malignant_end_to_end": {"macro_f1": 0.88, "accuracy": 0.99},
        "overall": {"macro_f1": 0.97, "accuracy": 1.0},
        "binary": {"macro_f1": 0.96, "accuracy": 1.0},
    }

    assert monitor_value(metrics, "malignant_macro_f1") == pytest.approx(0.37)


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        [],
        {},
        {"malignant": None},
        {"malignant": {}},
        {"malignant": {"accuracy": 0.9}},
    ],
)
def test_primary_monitor_rejects_missing_metrics_sections_and_keys(metrics):
    """主监控必须以中文 ValueError 拒绝缺失、空或非法指标结构。"""
    with pytest.raises(ValueError) as error:
        monitor_value(metrics, "malignant_macro_f1")

    _assert_chinese_value_error(error)


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        "0.4",
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_primary_monitor_rejects_bool_non_numeric_and_non_finite_values(value):
    """主监控值必须是非布尔且有限的实数。"""
    metrics = {"malignant": {"macro_f1": value}}

    with pytest.raises(ValueError) as error:
        monitor_value(metrics, "malignant_macro_f1")

    _assert_chinese_value_error(error)


@pytest.mark.parametrize(
    "monitor_metric, metrics",
    [
        ("macro_f1", {"malignant": {"accuracy": 0.8}}),
        ("balanced_acc", {"malignant": {"macro_f1": 0.8}}),
        ("acc", {"overall": {}}),
    ],
)
def test_diagnostic_monitor_rejects_missing_metric_keys(
    monitor_metric,
    metrics,
):
    """显式诊断模式也必须以中文 ValueError 拒绝缺失指标键。"""
    with pytest.raises(ValueError) as error:
        monitor_value(
            metrics,
            monitor_metric,
            allow_diagnostic=True,
        )

    _assert_chinese_value_error(error)


def test_unknown_monitor_metric_raises_chinese_value_error():
    """未知监控指标必须以中文 ValueError 明确拒绝。"""
    with pytest.raises(ValueError) as error:
        monitor_value({"malignant": {"macro_f1": 0.4}}, "flat5_accuracy")

    message = str(error.value)
    assert "monitor_metric" in message
    _assert_chinese_value_error(error)


def test_cli_defaults_to_reliable_flat4_configuration():
    """统一入口默认选择论文主模型需要的可靠 flat4 配置。"""
    parser = build_parser()

    args = parser.parse_args(["--pretrained_path", "TinyUSFM.pth"])

    assert args.task_mode == "flat4"
    assert args.lambda_bm == 0.3
    assert args.monitor_metric == "malignant_macro_f1"
    assert args.beta == 0.0
    assert args.gamma == 0.0
    assert args.label_smoothing == 0.0


@pytest.mark.parametrize("flag", ["--no_augment", "--no-augment"])
def test_overfit_cli_accepts_both_no_augment_spellings(flag):
    """下划线和连字符写法必须汇聚到同一个 augment 字段。"""
    parser = build_parser()

    args = parser.parse_args([
        "--pretrained_path",
        "TinyUSFM.pth",
        "--task_mode",
        "overfit",
        "--overfit_samples",
        "32",
        flag,
    ])

    assert args.overfit_samples == 32
    assert args.augment is False
    assert "no_augment" not in vars(args)


def test_cli_reports_invalid_arguments_in_chinese(capsys):
    """非法 choice 必须翻译正文并保留参数、非法值和允许值。"""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--pretrained_path",
            "TinyUSFM.pth",
            "--task_mode",
            "unknown",
        ])

    stderr = capsys.readouterr().err
    assert "参数错误" in stderr
    assert "error:" not in stderr
    assert "--task_mode" in stderr
    assert "unknown" in stderr
    assert "overfit" in stderr
    assert "flat4" in stderr
    assert "invalid choice" not in stderr
    assert "choose from" not in stderr


def test_cli_reports_missing_required_argument_in_chinese(capsys):
    """缺少必需参数必须使用中文正文并保留参数名。"""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    stderr = capsys.readouterr().err
    assert "参数错误" in stderr
    assert "缺少必需参数" in stderr
    assert "--pretrained_path" in stderr
    assert "required" not in stderr


@pytest.mark.parametrize(
    "arguments, expected_fragments, forbidden_fragment",
    [
        (
            ["--pretrained_path", "TinyUSFM.pth", "--unknown", "value"],
            ["无法识别参数", "--unknown", "value"],
            "unrecognized arguments",
        ),
        (
            ["--pretrained_path", "TinyUSFM.pth", "--epochs", "many"],
            ["--epochs", "many", "有效整数"],
            "invalid int value",
        ),
    ],
)
def test_cli_translates_other_common_argument_errors(
    capsys,
    arguments,
    expected_fragments,
    forbidden_fragment,
):
    """未知参数和数值类型错误也必须保留中文线索。"""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(arguments)

    stderr = capsys.readouterr().err
    for fragment in expected_fragments:
        assert fragment in stderr
    assert forbidden_fragment not in stderr


@pytest.mark.parametrize(
    "overrides, expected_fragment",
    [
        ({"beta": 0.1}, "beta"),
        ({"gamma": 0.1}, "gamma"),
        ({"label_smoothing": 0.1}, "label_smoothing"),
        ({"lambda_bm": 0.2}, "lambda_bm"),
        ({"monitor_metric": "acc"}, "monitor_metric"),
    ],
)
def test_reliable_configuration_rejects_hidden_objective_changes(
    overrides,
    expected_fragment,
):
    """可靠基线必须拒绝会改变目标函数或模型选择口径的参数。"""
    args = build_parser().parse_args(["--pretrained_path", "TinyUSFM.pth"])
    for name, value in overrides.items():
        setattr(args, name, value)

    with pytest.raises(ValueError) as error:
        validate_reliable_configuration(args)

    assert expected_fragment in str(error.value)
    _assert_chinese_value_error(error)


def _criterion_samples():
    return [
        {"class_label": 0, "malignancy_label": 1, "subtype_label": 0},
        {"class_label": 0, "malignancy_label": 1, "subtype_label": 0},
        {"class_label": 1, "malignancy_label": 1, "subtype_label": 1},
        {"class_label": 4, "malignancy_label": 0, "subtype_label": -1},
    ]


@pytest.mark.parametrize(
    "task_mode, expected_keys, expected_weight_count",
    [
        ("overfit", {"class"}, None),
        ("flat4", {"class"}, 4),
        ("flat5", {"class"}, 5),
        ("dual_head", {"malignancy", "subtype"}, None),
    ],
)
def test_criteria_follow_each_task_mode(
    task_mode,
    expected_keys,
    expected_weight_count,
):
    """四种任务模式必须使用各自标签空间的交叉熵。"""
    samples = _criterion_samples()
    if task_mode == "flat4":
        samples = [
            sample for sample in samples if sample["subtype_label"] != -1
        ]
    criteria = build_criteria_for_mode(
        task_mode,
        samples,
        torch.device("cpu"),
    )

    assert set(criteria) == expected_keys
    if expected_weight_count is not None:
        assert len(criteria["class"].weight) == expected_weight_count
    if task_mode == "overfit":
        assert criteria["class"].weight is None
    if task_mode == "dual_head":
        assert len(criteria["malignancy"].weight) == 2
        assert len(criteria["subtype"].weight) == 4
        assert criteria["malignancy"].weight.tolist() != (
            criteria["subtype"].weight[:2].tolist()
        )


class _SamplesOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


@pytest.mark.parametrize(
    "task_mode, expected_weights",
    [
        ("flat4", [0.5, 0.5, 1.0]),
        ("flat5", [0.5, 0.5, 1.0]),
        ("dual_head", [0.5, 0.5, 1.0]),
    ],
)
def test_balanced_sampler_uses_the_training_task_label(
    task_mode,
    expected_weights,
):
    """平衡采样必须按当前训练目标标签计算逆频率。"""
    samples = [
        {"subtype_label": 0, "class_label": 0},
        {"subtype_label": 0, "class_label": 0},
        {"subtype_label": 1, "class_label": 4},
    ]
    args = Namespace(
        sampler="balanced",
        batch_size=2,
        num_workers=0,
    )

    loader = build_train_loader(_SamplesOnlyDataset(samples), task_mode, args)

    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert loader.sampler.weights.tolist() == expected_weights


def test_unbalanced_loader_shuffles_and_overfit_rejects_balanced_sampler():
    """none 使用随机打乱，已均衡的过拟合子集拒绝二次平衡。"""
    dataset = _SamplesOnlyDataset([
        {"subtype_label": 0, "class_label": 0},
        {"subtype_label": 1, "class_label": 1},
    ])
    args = Namespace(sampler="none", batch_size=2, num_workers=0)

    loader = build_train_loader(dataset, "overfit", args)

    assert isinstance(loader.sampler, RandomSampler)
    args.sampler = "balanced"
    with pytest.raises(ValueError) as error:
        build_train_loader(dataset, "overfit", args)
    _assert_chinese_value_error(error)


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.patch_embed = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(2)
        for parameter in self.blocks[0].parameters():
            parameter.requires_grad = False


class _TinyTextBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Linear(2, 2)
        self.projection = nn.Linear(2, 2)
        for parameter in self.bert.parameters():
            parameter.requires_grad = False


class _TinyStage2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.task_heads = nn.Linear(2, 2)
        self.ip_adapters = nn.ModuleDict({"1": nn.Linear(2, 2)})
        self.cross_attention = nn.Linear(2, 2)
        self.cdfi_branch = nn.Linear(2, 2)
        self.encoder = _TinyEncoder()
        self.text_branch = _TinyTextBranch()


def test_optimizer_covers_every_trainable_parameter_once():
    """优化器参数组必须覆盖全部可训练参数且没有重复。"""
    model = _TinyStage2Model()

    optimizer = build_optimizer(model, base_lr=1e-3, weight_decay=0.05)

    optimized = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert {id(parameter) for parameter in optimized} == {
        id(parameter) for parameter in expected
    }
    assert len(optimized) == len({id(parameter) for parameter in optimized})
    assert {group["name"] for group in optimizer.param_groups} == {
        "task_heads",
        "ip_adapters",
        "cross_attention",
        "cdfi_branch",
        "encoder_block_1",
        "patch_embed_norm",
        "text_projection",
    }


@pytest.mark.parametrize(
    "history, prediction_distribution",
    [
        ([{"train": {"accuracy": 0.97, "total_loss": 0.01}}], [8, 8, 8, 8]),
        ([{"train": {"accuracy": 1.0, "total_loss": float("nan")}}], [8, 8, 8, 8]),
        ([{"train": {"accuracy": 1.0, "total_loss": 0.01}}], [16, 16, 0, 0]),
    ],
)
def test_overfit_gate_rejects_each_failure_condition(
    history,
    prediction_distribution,
):
    """门禁必须同时满足准确率、有限损失和四类预测齐全。"""
    metrics = {
        "malignant": {
            "prediction_distribution": prediction_distribution,
        }
    }

    with pytest.raises(RuntimeError) as error:
        assert_overfit_gate(32, history, metrics)

    _assert_chinese_value_error(error)


def test_overfit_gate_accepts_complete_four_class_fit():
    """满足全部约束的过拟合事实必须通过门禁。"""
    history = [{"train": {"accuracy": 0.98, "total_loss": 0.001}}]
    metrics = {"malignant": {"prediction_distribution": [8, 8, 8, 8]}}

    assert_overfit_gate(32, history, metrics)


def test_restore_manifest_splits_preserves_exact_names_and_case_order():
    """复评必须按 manifest 原名和 case ID 顺序精确恢复。"""
    canonical = {
        "train": [{"case_id": "A"}, {"case_id": "B"}],
        "malignant_test": [{"case_id": "C"}, {"case_id": "A"}],
    }
    manifest = {
        "splits": {
            "train": ["B", "A"],
            "malignant_test": ["C"],
        }
    }

    restored = restore_manifest_splits(canonical, manifest)

    assert list(restored) == ["train", "malignant_test"]
    assert [sample["case_id"] for sample in restored["train"]] == ["B", "A"]
    assert [sample["case_id"] for sample in restored["malignant_test"]] == ["C"]


def test_checkpoint_evaluation_rejects_task_mode_mismatch_before_model_build(
    tmp_path,
):
    """复评必须先校验检查点训练模式，禁止用当前模式静默覆盖。"""
    checkpoint_path = tmp_path / "best_model.pth"
    checkpoint_path.touch()
    save_json(tmp_path / "args.json", {
        "task_mode": "dual_head",
        "img_size": 224,
        "max_text_len": 128,
    })
    save_json(tmp_path / "split_manifest.json", {
        "task_mode": "dual_head",
        "splits": {},
    })
    args = Namespace(
        evaluate_checkpoint=str(checkpoint_path),
        task_mode="flat4",
        eval_output=str(tmp_path / "metrics.json"),
    )

    with pytest.raises(ValueError) as error:
        evaluate_checkpoint(args, torch.device("cpu"), {})

    assert "task_mode" in str(error.value)
    _assert_chinese_value_error(error)


def _checkpoint_cli_args(checkpoint_path):
    return build_parser().parse_args([
        "--pretrained_path",
        "TinyUSFM.pth",
        "--task_mode",
        "flat4",
        "--evaluate_checkpoint",
        str(checkpoint_path),
    ])


def test_checkpoint_main_builds_splits_from_saved_metadata(
    tmp_path,
    monkeypatch,
):
    """复评 canonical 划分必须使用历史运行保存的自定义元数据。"""
    checkpoint_path = tmp_path / "best_model.pth"
    checkpoint_path.touch()
    save_json(tmp_path / "args.json", {
        "task_mode": "flat4",
        "malignant_metadata": "历史恶性.csv",
        "benign_metadata": "历史良性.csv",
    })
    save_json(tmp_path / "split_manifest.json", {
        "task_mode": "flat4",
        "splits": {},
    })
    canonical_splits = {"malignant_test": [{"case_id": "历史病例"}]}

    def fake_build_fair_splits(
        project_root,
        task_mode,
        malignant_metadata,
        benign_metadata,
    ):
        assert project_root == train_stage2_module.PROJECT_ROOT
        assert task_mode == "flat4"
        assert malignant_metadata == "历史恶性.csv"
        assert benign_metadata == "历史良性.csv"
        return canonical_splits

    def fake_evaluate_checkpoint(args, device, received_splits):
        assert received_splits is canonical_splits
        return {"malignant": {"macro_f1": 0.5}}

    monkeypatch.setattr(
        train_stage2_module,
        "build_fair_splits",
        fake_build_fair_splits,
    )
    monkeypatch.setattr(
        train_stage2_module,
        "evaluate_checkpoint",
        fake_evaluate_checkpoint,
    )

    result = train_stage2_module.main(
        _checkpoint_cli_args(checkpoint_path)
    )

    assert result == {"malignant": {"macro_f1": 0.5}}


def test_checkpoint_main_validates_saved_mode_before_building_splits(
    tmp_path,
    monkeypatch,
):
    """保存 task_mode 不一致时不得先读取当前 CLI 对应的数据。"""
    checkpoint_path = tmp_path / "best_model.pth"
    checkpoint_path.touch()
    save_json(tmp_path / "args.json", {
        "task_mode": "dual_head",
        "malignant_metadata": "历史恶性.csv",
        "benign_metadata": "历史良性.csv",
    })
    save_json(tmp_path / "split_manifest.json", {
        "task_mode": "dual_head",
        "splits": {},
    })

    def fail_if_splits_are_built(*args, **kwargs):
        raise AssertionError("task_mode 校验前不应构造 canonical 划分")

    monkeypatch.setattr(
        train_stage2_module,
        "build_fair_splits",
        fail_if_splits_are_built,
    )

    with pytest.raises(ValueError) as error:
        train_stage2_module.main(
            _checkpoint_cli_args(checkpoint_path)
        )

    assert "task_mode" in str(error.value)
    _assert_chinese_value_error(error)


def test_overfit_gate_success_is_persisted_as_strict_json(tmp_path):
    """过拟合门禁成功结论必须写入稳定的严格 JSON 结构。"""
    history = [{"train": {"accuracy": 0.98, "total_loss": 0.001}}]
    metrics = {"malignant": {"prediction_distribution": [8, 8, 8, 8]}}

    result = train_stage2_module.enforce_overfit_gate(
        tmp_path,
        32,
        history,
        metrics,
    )

    raw = (tmp_path / "overfit_gate.json").read_text(encoding="utf-8")
    assert json.loads(raw) == result == {
        "passed": True,
        "sample_count": 32,
        "final_accuracy": 0.98,
        "final_total_loss": 0.001,
        "prediction_distribution": [8, 8, 8, 8],
    }


def test_overfit_gate_failure_is_persisted_before_runtime_error(tmp_path):
    """非有限损失失败时必须先保存严格 JSON，再抛中文异常。"""
    history = [{
        "train": {
            "accuracy": 1.0,
            "total_loss": float("nan"),
        }
    }]
    metrics = {"malignant": {"prediction_distribution": [8, 8, 8, 8]}}

    with pytest.raises(RuntimeError) as error:
        train_stage2_module.enforce_overfit_gate(
            tmp_path,
            32,
            history,
            metrics,
        )

    raw = (tmp_path / "overfit_gate.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload == {
        "passed": False,
        "sample_count": 32,
        "final_accuracy": 1.0,
        "final_total_loss": "nan",
        "prediction_distribution": [8, 8, 8, 8],
    }
    assert "NaN" not in raw
    _assert_chinese_value_error(error)


class _MainLoopOverfitDataset(torch.utils.data.Dataset):
    def __init__(self, project_root, samples, img_size):
        self.samples = [dict(sample) for sample in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "case_id": sample["case_id"],
            "bus_img": torch.zeros(1, 2, 2),
            "subtype_label": sample["subtype_label"],
        }


def test_main_persists_failed_gate_before_strict_history_rejects_nan(
    tmp_path,
    monkeypatch,
):
    """主循环遇到 NaN 损失时必须先写失败门禁，再抛门禁异常。"""
    samples = [
        {
            "case_id": f"{label}-{index}",
            "class_label": label,
            "malignancy_label": 1,
            "subtype_label": label,
            "split": "train",
            "source": "malignant",
        }
        for label in range(4)
        for index in range(8)
    ]
    monkeypatch.setattr(
        train_stage2_module,
        "build_fair_splits",
        lambda *args, **kwargs: {"train": samples},
    )
    monkeypatch.setattr(
        train_stage2_module,
        "BUSOverfitDataset",
        _MainLoopOverfitDataset,
    )
    monkeypatch.setattr(
        train_stage2_module,
        "build_model",
        lambda args, device: nn.Linear(1, 1).to(device),
    )
    monkeypatch.setattr(
        train_stage2_module,
        "train_one_epoch",
        lambda *args, **kwargs: {
            "total_loss": float("nan"),
            "accuracy": 1.0,
        },
    )
    monkeypatch.setattr(
        train_stage2_module,
        "evaluate_loader",
        lambda *args, **kwargs: {
            "malignant": {
                "prediction_distribution": [8, 8, 8, 8],
            }
        },
    )
    args = build_parser().parse_args([
        "--pretrained_path",
        "TinyUSFM.pth",
        "--task_mode",
        "overfit",
        "--epochs",
        "1",
        "--num_workers",
        "0",
        "--output_root",
        str(tmp_path),
    ])

    with pytest.raises(RuntimeError) as error:
        train_stage2_module.main(args)

    run_dirs = list(tmp_path.glob("overfit_*"))
    assert len(run_dirs) == 1
    gate_path = run_dirs[0] / "overfit_gate.json"
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["sample_count"] == 32
    assert payload["final_accuracy"] == 1.0
    assert payload["final_total_loss"] == "nan"
    assert payload["prediction_distribution"] == [8, 8, 8, 8]
    assert not (run_dirs[0] / "history.json").exists()
    _assert_chinese_value_error(error)
